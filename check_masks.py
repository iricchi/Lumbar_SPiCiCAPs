import os
import sys
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.append('.')
from src.paths import PAM50_LUMBAR_PATH, LUMBAR_TEMPLATE, SPINAL_LEVELS

PATH = Path('/media')/'miplab-nas2'/'Data2'/'SpinalCord'/'Spinal_fMRI_Lumbar'
SUBJECTS = ['LU_AT','LU_EP','LU_FB', 'LU_GL', 'LU_GP', 'LU_MD', 'LU_MP', 'LU_SA', 'LU_SL',
            'LU_VG','LU_YF','LU_EM', 'LU_SM', 'LU_ML', 'LU_NS','LU_BN', 'LU_NB']
WARP_PATH = Path('func')/'Normalization'/'warp_fmri2template_2.nii.gz'
MASK_PATH = Path('func')/'Segmentation'/'mask_sco.nii.gz'



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()

    norm_path = Path('normalized_masks')
    if args.normalize:
        norm_path.mkdir(parents=True, exist_ok=True)

        # Apply normalization to all masks
        for subject in SUBJECTS:
            mask_path = PATH/subject/MASK_PATH
            warp_path = PATH/subject/WARP_PATH
            out_path = norm_path/f'{subject}_mask.nii.gz'
            cmd = f'sct_apply_transfo -i {mask_path} -d {PAM50_LUMBAR_PATH} -w {warp_path} -x nn -o {out_path}'
            os.system(cmd)

    # For each level, count the number of voxels in template
    original_levels = nib.load(LUMBAR_TEMPLATE).get_fdata()
    vals, counts = np.unique(original_levels, return_counts=True)
    vals = vals.astype(int)

    level_names = SPINAL_LEVELS[vals]
    l2_idx = np.where(level_names=='L2')[0][0]
    l2_total_count = counts[l2_idx]

    percs = []
    for subject in SUBJECTS:
        mask_path = norm_path/f'{subject}_mask.nii.gz'
        mask = nib.load(mask_path).get_fdata()
        masked_levels = original_levels * mask
        vals, counts = np.unique(masked_levels, return_counts=True)
        vals = vals.astype(int)
        level_names = SPINAL_LEVELS[vals]

        # Find L2
        l2_idx = np.where(level_names=='L2')[0]
        if len(l2_idx)==0:
            l2_count = 0
        else:
            l2_count = counts[l2_idx[0]]
    
        percs.append(l2_count/l2_total_count)
        print(f'{subject}: {l2_count/l2_total_count:.2f}')

    f, ax = plt.subplots(figsize=(10, 5))
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Subject')
    ax.set_title('Percentage of L2 voxels in mask')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Rotate xticks 45 degrees
    plt.xticks(rotation=45)
    plt.bar(SUBJECTS, percs)
    plt.savefig('./L2.png')
