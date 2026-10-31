# dataset/cervix_dataset.py
#
# Reads patches from the per-patient HDF5 files written by extract_patches.py.
#
# Each patch exists on disk exactly once, keyed by (patient_id, slide_idx, r,
# c). The model needs a target patch plus its 8 spatial neighbours per
# forward pass (methodology.tex section on spatial extraction) — this class
# reconstructs that neighbourhood at __getitem__ time by looking up (r+dr,
# c+dc) WITHIN THE SAME SLIDE, rather than the neighbours ever being written
# to disk a second time. Storing them redundantly would multiply extraction
# time and disk usage by ~9x for data that already exists elsewhere as some
# other tile's target patch.
#
# slide_idx matters for real: a patient with multiple diagnostic slides has
# their tiles indexed r,c independently per slide — two different slides'
# grids are two unrelated coordinate systems. Indexing only by (patient_id,
# r, c) would look up (r=3,c=4) on slide 0 and (r=3,c=4) on slide 1 as if
# they were "the same position" and could return a neighbour from a
# physically unrelated region of tissue. This affects a small fraction of
# patients (multi-slide is uncommon) but silently, so it's handled correctly
# rather than assumed away.
#
# Missing neighbours (slide boundary, or filtered out by the tissue-fraction
# threshold in Stage 1) are zero-padded, matching methodology.tex: "Patches at
# slide boundaries where one or more neighbours fall outside the tissue
# region are zero-padded, and the SN-ViT module learns to discount
# zero-padded neighbour positions through the attention mechanism."
#
# All array layout conversions go through utils.array_ops — see that module
# for why this matters.

import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.array_ops import assert_valid_patch  # noqa: E402

NEIGHBOUR_OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if (dr, dc) != (0, 0)]  # 8 offsets, target excluded


class CervixDataset(Dataset):
    def __init__(self, metadata_path, h5_dir, patient_ids, grade_lookup=None,
                 patch_size=256, transform=None):
        """
        metadata_path : path to patch_metadata.parquet (from extract_patches.py)
        h5_dir        : directory containing per-patient .h5 files
        patient_ids   : patients to include (i.e. this split's train/val/test IDs)
        grade_lookup  : dict patient_id -> int grade in {0,1,2}, or -1/absent for GX
        """
        df = pd.read_parquet(metadata_path)
        df = df[df['patient_id'].isin(set(patient_ids))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError('no patches found for the given patient_ids — '
                             'check that metadata_path and patient_ids agree')
        self.df = df
        self.h5_dir = h5_dir
        self.patch_size = patch_size
        self.transform = transform
        self.grade_lookup = grade_lookup or {}

        # (patient_id, slide_idx, r, c) -> row index, for O(1) neighbour
        # lookup scoped to the correct slide. Falls back to slide_idx=0 for
        # metadata written before this fix (single-slide-per-patient only,
        # where the distinction was moot).
        has_slide_idx = 'slide_idx' in df.columns
        self._index = {}
        for i, row in df.iterrows():
            sidx = int(row['slide_idx']) if has_slide_idx else 0
            self._index[(row['patient_id'], sidx, int(row['r']), int(row['c']))] = i

        self._open_files = {}  # lazy per-worker file handles

    def __len__(self):
        return len(self.df)

    def _h5(self, patient_id, h5_path):
        # Each DataLoader worker process gets its own handle (opened lazily on
        # first access), since h5py file handles are not fork-safe to share.
        key = (os.getpid(), patient_id)
        if key not in self._open_files:
            self._open_files[key] = h5py.File(
                os.path.join(self.h5_dir, h5_path), 'r')
        return self._open_files[key]

    def _load_patch(self, patient_id, h5_path, h5_key):
        f = self._h5(patient_id, h5_path)
        arr = f['patches'][h5_key][:]
        assert_valid_patch(arr, expect='chw')
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id, r, c = row['patient_id'], int(row['r']), int(row['c'])
        slide_idx = int(row['slide_idx']) if 'slide_idx' in row.index else 0

        target = self._load_patch(patient_id, row['h5_path'], row['h5_key'])

        neighbours = np.zeros((8, 3, self.patch_size, self.patch_size),
                              dtype=np.uint8)
        for n, (dr, dc) in enumerate(NEIGHBOUR_OFFSETS):
            key = (patient_id, slide_idx, r + dr, c + dc)
            if key in self._index:
                nrow = self.df.iloc[self._index[key]]
                neighbours[n] = self._load_patch(patient_id, nrow['h5_path'],
                                                 nrow['h5_key'])
            # else: stays zero — boundary or filtered-out neighbour

        target_t = torch.from_numpy(target).float() / 255.0
        neighbours_t = torch.from_numpy(neighbours).float() / 255.0
        position = torch.tensor([c, r], dtype=torch.float32)  # (x, y) order

        if self.transform is not None:
            target_t = self.transform(target_t)
            neighbours_t = torch.stack([self.transform(neighbours_t[i])
                                        for i in range(8)])

        saliency_label = torch.tensor(row['saliency_label'], dtype=torch.float32)
        grade = self.grade_lookup.get(patient_id, -1)
        grade_label = torch.tensor(grade, dtype=torch.long)

        return {
            'patch': target_t, 'neighbours': neighbours_t, 'position': position,
            'saliency_label': saliency_label, 'grade_label': grade_label,
            'patient_id': patient_id, 'r': r, 'c': c,
        }

    def __del__(self):
        for f in getattr(self, '_open_files', {}).values():
            try:
                f.close()
            except Exception:
                pass
