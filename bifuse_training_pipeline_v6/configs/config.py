import os


class Config:
    """
    CHANGED from the recovered config:
      - Removed a hardcoded Hugging Face API token that was committed in
        plaintext. Never put credentials in a config file — if that token
        was real, revoke it and issue a fresh one.
      - Removed CONCH/UNI/Colab paths — out of scope for this rebuttal
        (ResNet-50 only, per the locked-in scope decision), deferred to the
        journal paper's backbone comparison.
      - Removed GRADE_WEIGHTS as a hardcoded constant. The recovered value
        ([1/0.073, 1/0.532, 1/0.395]) didn't match ANY real distribution this
        project has ever measured — not the fabricated ~15/35/50 estimate,
        not the real 6.3/46.5/35.7 from patient_split.json. This is exactly
        the "grade weights computed at runtime, never hardcoded" bug flagged
        early in the rebuild. train.py computes this from the actual train
        split, every run, from whatever patient_split.json is loaded.
      - Split-file key names fixed to match build_split.py's real output
        (train_patients/val_patients/test_patients, not train/val/test) —
        the old config would have KeyError'd, or worse, silently fallen back
        to a tiny 4-patient hardcoded example list if that lookup were ever
        wrapped in a try/except.
    """
    BASE = r'C:\Users\himohamm\Documents\BiFuse_CervNet'

    SPLIT_FILE     = os.path.join(BASE, 'patient_split.json')
    METADATA_FILE  = os.path.join(BASE, 'patches', 'patch_metadata.parquet')
    H5_DIR         = os.path.join(BASE, 'patches')
    CHECKPOINTS_DIR = os.path.join(BASE, 'checkpoints')
    LOGS_DIR        = os.path.join(BASE, 'logs')

    PATCH_SIZE   = 256
    K_NEIGHBOURS = 8
    PATCHES_PER_PATIENT = 100  # epoch-level subsampling, per methodology
    VAL_PATCHES_PER_PATIENT = 200  # fixed once per run, NOT resampled per
    # epoch — val needs to stay the same set across epochs for early-stopping
    # comparisons to mean anything. Larger than train's 100 since val forward
    # passes are cheaper (no backward pass) and a bit more val data gives
    # steadier metrics.

    CNN_OUT_DIM        = 512
    VIT_DIM            = 256
    VIT_HEADS          = 8
    VIT_LAYERS         = 4
    FUSION_DIM         = 256
    DROPOUT            = 0.4
    NUM_CLASSES_DETECT = 1
    NUM_CLASSES_GRADE  = 3
    SEED = 42

    # --- gate configuration ---
    GATE_MODE          = 'spatial'   # 'spatial' | 'scalar' | 'identity'
    GATE_NORMALIZE     = True
    GATE_USE_ATTENTION = True
    SPATIAL_GRID_H      = 8
    SPATIAL_GRID_W      = 8

    EPOCHS       = 50
    PATIENCE     = 10
    BATCH_SIZE   = 128        # 51GB VRAM gives real headroom to raise this
    NUM_WORKERS  = 4
    LR           = 1e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP    = 1.0

    LAMBDA_DETECT = 0.6
    LAMBDA_GRADE  = 0.4

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD  = [0.229, 0.224, 0.225]


def get_config():
    cfg = Config()
    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR, exist_ok=True)
    return cfg
