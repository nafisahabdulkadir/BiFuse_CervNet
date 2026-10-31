# utils/tcga.py
#
# TCGA case-ID parsing lived as two separate inline implementations —
# find_patient_id() in build_tissue_coords.py and a reimplementation inside
# extract_patches.py's svs_for(). Two copies of "the same" logic is exactly
# the pattern that let the original transpose bug diverge silently. One
# function, imported everywhere.

def find_patient_id(filename):
    """TCGA filenames encode the case ID as the first 3 hyphenated fields.

    e.g. 'TCGA-XX-0001-01Z-00-DX1.svs' -> 'TCGA-XX-0001'
    """
    parts = filename.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    import os
    return os.path.splitext(filename)[0]
