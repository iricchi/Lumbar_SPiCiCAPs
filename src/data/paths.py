import numpy as np
from pathlib import Path


LUMBAR_RAW_DATA_PATH = Path('/media')/'miplab-nas2'/'Data2'/'SpinalCord'/'Spinal_fMRI_Lumbar'
CERVICAL_PATH = Path('/media')/'miplab-nas2'/'Data2'/'SpinalCord'/'4_StudentProjects'/'Daniel'/'cervical'/'SPiCiCAPs'
#LUMBAR_PATH = Path('/media')/'miplab-nas2'/'Data2'/'SpinalCord'/'4_StudentProjects'/'Daniel'/'lumbar'/'PAM50_all_Alpha_5_950DOT05'
#LUMBAR_PATH = Path('/media')/'miplab-nas2'/'Data2'/'SpinalCord'/'Spinal_fMRI_Lumbar'/'iCAPs_results'/'PAM50_all_Alpha_5_950DOT05'
LUMBAR_PATH = Path('/media')/'miplab-nas2'/'Data'/'SPiCiCAP_Ila'/'Lumbar'/'iCAPs_results'/'PAM50_all_Alpha_5_950DOT05'


ORIGINAL_TEMPLATE = Path('templates')/'PAM50_spinal_levels_original.nii.gz'
CERVICAL_TEMPLATE = Path('templates')/'PAM50_spinal_levels.nii.gz'
LUMBAR_TEMPLATE = Path('templates')/'PAM50_spinal_levels_lumbar.nii.gz'

PAM50_CERVICAL_PATH = Path('/media')/'miplab-nas2'/'Data2'/'SpinalCord'/'4_StudentProjects'/'Daniel'/'cervical'/'template'/'PAM50_t2.nii.gz'
PAM50_LUMBAR_PATH   = Path('/media')/'miplab-nas2'/'Data2'/'SpinalCord'/'4_StudentProjects'/'Daniel'/'lumbar'/'templates'/'PAM50_t2_lumbar_crop.nii.gz'


CERVICAL_LEVELS = np.array(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                            'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12'])

LUMBAR_LEVELS = np.array(['T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                          'L1', 'L2', 'L3', 'L4', 'L5',
                          'S1', 'S2', 'S3', 'S4', 'S5'])

SPINAL_LEVELS = np.array(['',   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 
                          'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                          'L1', 'L2', 'L3', 'L4', 'L5',
                          'S1', 'S2', 'S3', 'S4', 'S5'])

LUMBAR_SUBJECTS = np.array(['LU_AT', 'LU_EP', 'LU_FB', 'LU_GL', 'LU_GP', 'LU_MD', 'LU_MP', 'LU_SA', 'LU_SL',
                            'LU_VG', 'LU_YF','LU_EM', 'LU_SM', 'LU_ML', 'LU_NS','LU_BN', 'LU_NB'])