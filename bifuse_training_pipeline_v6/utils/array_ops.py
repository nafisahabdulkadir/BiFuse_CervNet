# utils/array_ops.py
#
# SINGLE source of truth for patch array layout conversion.
#
# The transpose bug survived a repair pass because the conversion was written
# out separately in cervix_dataset.py and dataloader.py, and one copy used
# transpose(0,2,1) while the other used transpose(2,0,1). Both "worked" on a
# (256, 3, 256) array without raising, so nothing failed loudly.
#
# Every call site must use these helpers. Do not inline .transpose() on patch
# arrays anywhere else.

import numpy as np

N_CHANNELS = 3


def is_hwc(arr):
    return arr.ndim == 3 and arr.shape[2] == N_CHANNELS


def is_chw(arr):
    return arr.ndim == 3 and arr.shape[0] == N_CHANNELS


def hwc_to_chw(arr):
    """(H, W, C) -> (C, H, W). Idempotent if already CHW."""
    if is_chw(arr) and not is_hwc(arr):
        return arr
    if not is_hwc(arr):
        raise ValueError(
            f'expected (H, W, {N_CHANNELS}), got {arr.shape}. '
            'A (W, C, H) array here means the patch was written with the wrong '
            'layout upstream; fix extraction rather than transposing again.')
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def chw_to_hwc(arr):
    """(C, H, W) -> (H, W, C). Idempotent if already HWC."""
    if is_hwc(arr) and not is_chw(arr):
        return arr
    if not is_chw(arr):
        raise ValueError(f'expected ({N_CHANNELS}, H, W), got {arr.shape}')
    return np.ascontiguousarray(arr.transpose(1, 2, 0))


def assert_valid_patch(arr, expect='chw'):
    """Loud check for use at dataset boundaries."""
    ok = is_chw(arr) if expect == 'chw' else is_hwc(arr)
    if not ok:
        raise ValueError(f'patch failed {expect.upper()} check: shape={arr.shape}')
    return arr
