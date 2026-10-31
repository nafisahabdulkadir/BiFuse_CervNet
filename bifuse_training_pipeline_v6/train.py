#!/usr/bin/env python3
"""train.py — Phase 2/3 training: the identity-vs-spatial gate control run.

Usage:
    python train.py --gate-mode identity --epochs 50 --out m4_identity
    python train.py --gate-mode spatial  --epochs 50 --out m4_spatial
    python train.py --gate-mode none     --epochs 50 --out m3
    python train.py --gate-mode identity --resume checkpoints/m4_identity/last.pt

M3 ('none') is the paper's "unidirectional CNN-to-ViT, no gate" baseline —
see models/bifuse_cervnet.py's module docstring for exactly what changes
architecturally and the honest caveat about reconstruction vs. original code.

M3 (no gate at all) isn't run from this script — it's the pre-existing
baseline architecture without the gate/gate_pool/gate_proj pathway. This
script covers the M4 family, which is what actually needed re-verification.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from configs.config import get_config
from models.bifuse_cervnet import BiFuseCervNet
from dataset.cervix_dataset import CervixDataset


def compute_split_hash(split_path):
    with open(split_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def get_git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return 'unknown (not a git repo, or git not installed)'


def compute_grade_weights(patient_grade, train_patients, num_classes):
    """Inverse-frequency weights from the ACTUAL train split, every run.
    Never hardcoded — the recovered config's GRADE_WEIGHTS constant matched
    no real distribution this project ever measured. This function is the
    fix, and it runs fresh every time train.py starts.
    """
    counts = Counter()
    for p in train_patients:
        g = patient_grade.get(p, -1)
        if g is not None and g >= 0:
            counts[g] += 1
    weights = torch.ones(num_classes)
    for c in range(num_classes):
        n = counts.get(c, 0)
        if n > 0:
            weights[c] = 1.0 / n
    weights = weights / weights.sum() * num_classes  # normalize, mean 1
    return weights, dict(counts)


class EpochSubsampler:
    """100 patches/patient/epoch (per methodology), a different random
    subset each epoch via a seed that varies with the epoch number. Built
    once per training run; indices_for_epoch() is cheap to call repeatedly.
    """
    def __init__(self, dataset, patches_per_patient, seed):
        self.patches_per_patient = patches_per_patient
        self.base_seed = seed
        self.by_patient = {}
        for i, pid in enumerate(dataset.df['patient_id'].values):
            self.by_patient.setdefault(pid, []).append(i)

    def indices_for_epoch(self, epoch):
        rng = np.random.default_rng(self.base_seed + epoch)
        idx = []
        for pid, rows in self.by_patient.items():
            k = min(self.patches_per_patient, len(rows))
            idx.extend(rng.choice(rows, size=k, replace=False).tolist())
        return idx


def compute_saliency_metrics(probs, labels):
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    preds = (probs >= 0.5).astype(int)
    auroc = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float('nan')
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    return {'auroc': float(auroc), 'f1': float(f1), 'sens': float(sens),
            'spec': float(spec), 'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)}


def compute_grade_metrics(preds, labels, num_classes):
    """QWK, macro-F1, per-class F1, confusion matrix — excluding
    ignore_index=-1 (GX/unknown) entirely, matching train-time loss masking.
    This is the piece that never existed before this rebuild — the paper's
    Grading Results section was a placeholder from day one; this is what
    fills it in, every run, for real.
    """
    from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
    mask = labels >= 0
    if mask.sum() == 0:
        return {'qwk': float('nan'), 'macro_f1': float('nan'),
                'per_class_f1': [], 'confusion_matrix': []}
    p, l = preds[mask], labels[mask]
    qwk = cohen_kappa_score(l, p, weights='quadratic')
    macro_f1 = f1_score(l, p, average='macro',
                        labels=list(range(num_classes)), zero_division=0)
    per_class_f1 = f1_score(l, p, average=None,
                            labels=list(range(num_classes)),
                            zero_division=0).tolist()
    cm = confusion_matrix(l, p, labels=list(range(num_classes))).tolist()
    return {'qwk': float(qwk), 'macro_f1': float(macro_f1),
            'per_class_f1': per_class_f1, 'confusion_matrix': cm}


def run_epoch(model, loader, optimizer, device, cfg, grade_weights, train, tag=''):
    model.train() if train else model.eval()
    detect_criterion = nn.BCEWithLogitsLoss()
    grade_criterion = nn.CrossEntropyLoss(weight=grade_weights.to(device),
                                          ignore_index=-1)

    total_loss = 0.0
    all_probs, all_sal_labels = [], []
    all_grade_preds, all_grade_labels = [], []

    n_batches = len(loader)
    if n_batches == 0:
        raise ValueError(
            f'{tag} loader has ZERO batches — dataset has {len(loader.dataset)} '
            f'samples but batch_size={cfg.BATCH_SIZE}'
            f'{" and drop_last=True drops the only partial batch" if train else ""}. '
            f'This should never happen with real data (real train split is '
            f'thousands of patches) — check BATCH_SIZE against the actual '
            f'dataset/subsample size if you see this.')
    n_seen = 0
    t0 = time.time()
    last_print = t0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            patch = batch['patch'].to(device)
            neighbours = batch['neighbours'].to(device)
            position = batch['position'].to(device)
            sal_label = batch['saliency_label'].to(device)
            grade_label = batch['grade_label'].to(device)

            det, grd = model(patch, neighbours, position)
            loss_detect = detect_criterion(det, sal_label)
            # CrossEntropyLoss(ignore_index=-1) returns NaN, not 0, when a
            # batch has ZERO valid (non-ignored) targets — confirmed
            # directly, and confirmed this actually happened on a real run:
            # one all-grade-unknown batch poisoned that epoch's val_loss via
            # plain summation, which in turn permanently broke early
            # stopping (nan is never "improved" over anything, so patience
            # silently counted down even while the model was still
            # improving). Guarding at the source, in both train and val,
            # rather than relying solely on the val-ordering fix above —
            # shuffling makes this rare, not impossible.
            has_valid_grade = (grade_label >= 0).any()
            if has_valid_grade:
                loss_grade = grade_criterion(grd, grade_label)
            else:
                loss_grade = torch.zeros((), device=device)
            loss = cfg.LAMBDA_DETECT * loss_detect + cfg.LAMBDA_GRADE * loss_grade

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item() * patch.size(0)
            n_seen += patch.size(0)
            all_probs.append(torch.sigmoid(det).detach().cpu().numpy())
            all_sal_labels.append(sal_label.detach().cpu().numpy())
            all_grade_preds.append(grd.argmax(dim=1).detach().cpu().numpy())
            all_grade_labels.append(grade_label.detach().cpu().numpy())

            now = time.time()
            if now - last_print > 10 or (i + 1) == n_batches:
                elapsed = now - t0
                rate = n_seen / elapsed if elapsed > 0 else 0
                eta = (n_batches - i - 1) * (elapsed / (i + 1)) if i > 0 else 0
                print(f'    {tag} batch {i+1}/{n_batches}  {n_seen} patches  '
                      f'loss={loss.item():.4f}  {elapsed:.0f}s elapsed  '
                      f'{rate:.0f} patches/s  ETA {eta:.0f}s', flush=True)
                last_print = now

    n = len(loader.dataset)
    avg_loss = total_loss / max(n, 1)
    probs = np.concatenate(all_probs)
    sal_labels = np.concatenate(all_sal_labels)
    grade_preds = np.concatenate(all_grade_preds)
    grade_labels = np.concatenate(all_grade_labels)

    sal_metrics = compute_saliency_metrics(probs, sal_labels)
    grade_metrics = compute_grade_metrics(grade_preds, grade_labels,
                                          cfg.NUM_CLASSES_GRADE)
    return avg_loss, sal_metrics, grade_metrics


@torch.no_grad()
def log_gate_diagnostics(model, loader, device, max_batches=5):
    """Called every epoch — spatial_std must be logged as a training-time
    CURVE, not checked once. The softmax gate can gradually decay toward
    uniform under weight decay (see spatial_attention_gate.py's own
    docstring on this); catching that drift requires watching it happen
    across epochs, not a single end-of-training snapshot.
    """
    model.eval()
    stats_accum = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        patch = batch['patch'].to(device)
        neighbours = batch['neighbours'].to(device)
        position = batch['position'].to(device)
        _, stats = model.get_gate_values(patch, neighbours, position)
        stats_accum.append(stats)
    if not stats_accum:
        return {}
    keys = stats_accum[0].keys()
    return {k: float(np.mean([s[k] for s in stats_accum])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate-mode', choices=['spatial', 'scalar', 'identity', 'none'],
                    default='spatial')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--out', default=None,
                    help='checkpoint subdirectory name; defaults to gate mode')
    ap.add_argument('--resume', default=None,
                    help='path to a checkpoint to resume from')
    a = ap.parse_args()

    cfg = get_config()
    cfg.GATE_MODE = a.gate_mode
    if a.epochs:
        cfg.EPOCHS = a.epochs
    if a.patience:
        cfg.PATIENCE = a.patience
    if a.batch_size:
        cfg.BATCH_SIZE = a.batch_size
    if a.lr:
        cfg.LR = a.lr

    run_name = a.out or ('m3' if a.gate_mode == 'none' else f'm4_{a.gate_mode}')
    ckpt_dir = os.path.join(cfg.CHECKPOINTS_DIR, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg.LOGS_DIR, f'{run_name}.jsonl')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if device.type != 'cuda':
        print('WARNING: no GPU detected — this will be extremely slow')

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    with open(cfg.SPLIT_FILE) as f:
        split = json.load(f)
    train_patients = split['train_patients']
    val_patients = split['val_patients']
    patient_grade = dict(split['patient_grade'])
    split_hash = compute_split_hash(cfg.SPLIT_FILE)
    print(f'split: {len(train_patients)} train / {len(val_patients)} val patients, '
          f'hash={split_hash}')

    grade_weights, grade_counts = compute_grade_weights(
        patient_grade, train_patients, cfg.NUM_CLASSES_GRADE)
    print(f'grade class weights (computed from THIS train split, not '
          f'hardcoded): {grade_weights.tolist()}  (counts: {grade_counts})')

    train_ds = CervixDataset(cfg.METADATA_FILE, cfg.H5_DIR, train_patients,
                             grade_lookup=patient_grade, patch_size=cfg.PATCH_SIZE)
    val_ds = CervixDataset(cfg.METADATA_FILE, cfg.H5_DIR, val_patients,
                           grade_lookup=patient_grade, patch_size=cfg.PATCH_SIZE)
    print(f'train dataset: {len(train_ds)} total patches '
          f'(subsampled to {cfg.PATCHES_PER_PATIENT}/patient/epoch)')
    print(f'val dataset: {len(val_ds)} total patches '
          f'(subsampled ONCE to {cfg.VAL_PATCHES_PER_PATIENT}/patient, fixed '
          f'across all epochs of this run)')

    train_sampler = EpochSubsampler(train_ds, cfg.PATCHES_PER_PATIENT, cfg.SEED)
    val_sampler = EpochSubsampler(val_ds, cfg.VAL_PATCHES_PER_PATIENT, cfg.SEED)
    val_indices = val_sampler.indices_for_epoch(0)  # single fixed draw, reused
    # every epoch — NOT re-drawn per epoch like train, since a moving val set
    # would make epoch-to-epoch loss/AUROC comparisons (and early stopping)
    # meaningless.
    #
    # Shuffled once, with a fixed seed, before use. Patches in the parquet
    # are stored contiguously per patient (extraction writes one patient at
    # a time), and with shuffle=False on the DataLoader, an unshuffled index
    # list means every batch — especially the last one — can end up
    # dominated by a single patient. If that patient happens to be
    # grade-unknown, CrossEntropyLoss(ignore_index=-1) returns NaN for a
    # batch with zero valid targets, which poisons the whole epoch's
    # reported val_loss and silently breaks early stopping (nan is never
    # "improved" over anything, so patience counts down even while the
    # model is genuinely getting better). Confirmed this is exactly what
    # happened on a real run, not a hypothetical.
    rng = np.random.default_rng(cfg.SEED)
    val_indices = rng.permutation(val_indices).tolist()
    val_subset = Subset(val_ds, val_indices)
    print(f'val subset actually used: {len(val_subset)} patches')

    model = BiFuseCervNet(cfg).to(device)
    n_gate_params = (sum(p.numel() for p in model.gate.parameters()
                        if p.requires_grad) if model.gate is not None else 0)
    print(f'gate_mode={cfg.GATE_MODE}  gate trainable params={n_gate_params}'
          f'{" (M3: no gate module at all)" if model.gate is None else ""}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                   weight_decay=cfg.WEIGHT_DECAY)

    start_epoch = 0
    best_val_loss = float('inf')
    best_val_auroc = -1.0  # AUROC in [0,1]; -1 guarantees the first epoch "improves"
    epochs_no_improve = 0
    if a.resume and os.path.exists(a.resume):
        ckpt = torch.load(a.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_val_auroc = ckpt.get('best_val_auroc', -1.0)
        if 'epochs_no_improve' in ckpt:
            epochs_no_improve = ckpt['epochs_no_improve']
        else:
            print('WARNING: this checkpoint predates epochs_no_improve being '
                  'saved — early-stopping patience is restarting from 0 rather '
                  'than continuing where it left off. Harmless if this run had '
                  'not yet gone through any non-improving epochs; otherwise '
                  'this run may train longer than the original patience '
                  'setting intended.')
        print(f'resumed from {a.resume} at epoch {start_epoch}, '
              f'epochs_no_improve={epochs_no_improve}')

    val_loader = DataLoader(val_subset, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS)

    for epoch in range(start_epoch, cfg.EPOCHS):
        t0 = time.time()
        epoch_indices = train_sampler.indices_for_epoch(epoch)
        train_loader = DataLoader(Subset(train_ds, epoch_indices),
                                  batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, drop_last=True)
        # drop_last=True: BatchNorm1d in the fusion head requires >1 sample
        # per batch when training (it needs real batch statistics). Without
        # this, a training run would crash outright if the subsampled patch
        # count for some epoch happens to leave exactly 1 sample in the
        # final batch — confirmed this is a real failure mode, not
        # hypothetical, while testing at a small batch size.

        print(f'\n=== epoch {epoch+1}/{cfg.EPOCHS} starting: '
              f'{len(epoch_indices)} train patches, {len(val_subset)} val patches ===',
              flush=True)

        train_loss, train_sal, train_grade = run_epoch(
            model, train_loader, optimizer, device, cfg, grade_weights,
            train=True, tag='[train]')
        val_loss, val_sal, val_grade = run_epoch(
            model, val_loader, optimizer, device, cfg, grade_weights,
            train=False, tag='[val]')
        gate_stats = log_gate_diagnostics(model, val_loader, device)

        elapsed = time.time() - t0
        record = {
            'epoch': epoch, 'elapsed_s': round(elapsed, 1),
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_saliency': train_sal, 'val_saliency': val_sal,
            'train_grade': train_grade, 'val_grade': val_grade,
            'gate_diagnostics': gate_stats,
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        print(f'[{epoch+1}/{cfg.EPOCHS}] ({elapsed:.0f}s) '
              f'train_loss={train_loss:.4f} val_loss={val_loss:.4f} '
              f'val_auroc={val_sal["auroc"]:.4f} val_qwk={val_grade["qwk"]:.4f} '
              f'gate_spatial_std={gate_stats.get("spatial_std", float("nan")):.4f}')

        # NaN > anything is always False in Python — same bug class as the
        # earlier val_loss issue, just for a different metric. Unlikely at
        # real scale (saliency is ~50/50 balanced by design, val subsample
        # is 200+/patient), but made explicit rather than left as a second
        # silent copy of a bug we already got bitten by once.
        val_auroc = val_sal['auroc']
        if math.isnan(val_auroc):
            print(f'WARNING: val_auroc is NaN this epoch (val set may lack '
                  f'both saliency classes) — treated as no improvement, '
                  f'best_val_auroc left unchanged rather than corrupted.')
            improved = False
        else:
            improved = val_auroc > best_val_auroc
        if improved:
            best_val_auroc = val_auroc
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ckpt_payload = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_auroc': best_val_auroc,
            'epochs_no_improve': epochs_no_improve,
            'val_loss': val_loss, 'val_saliency': val_sal, 'val_grade': val_grade,
            'gate_diagnostics': gate_stats,
            'config': {k: v for k, v in vars(type(cfg)).items()
                      if not k.startswith('_') and not callable(v)},
            'grade_class_weights': grade_weights.tolist(),
            'split_file_hash': split_hash,
            'git_commit': get_git_commit(),
            'gate_mode': cfg.GATE_MODE,
        }
        torch.save(ckpt_payload, os.path.join(ckpt_dir, 'last.pt'))
        if improved:
            torch.save(ckpt_payload, os.path.join(ckpt_dir, 'best.pt'))

        if epochs_no_improve >= cfg.PATIENCE:
            print(f'early stopping at epoch {epoch+1} '
                  f'({cfg.PATIENCE} epochs without improvement)')
            break

    print(f'\ndone. best val_auroc={best_val_auroc:.4f} (val_loss at that '
          f'epoch: {best_val_loss:.4f}). '
          f'checkpoints in {ckpt_dir}, log at {log_path}')


if __name__ == '__main__':
    main()
