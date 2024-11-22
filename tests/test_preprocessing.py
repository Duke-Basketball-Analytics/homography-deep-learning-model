import numpy as np
def mask_dimensions():
    mask = np.load("../DL_masks/OFFENSE-40_richmond/Frame_300.npy")
    print(mask.shape)

mask_dimensions()