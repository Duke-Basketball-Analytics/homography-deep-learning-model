import numpy as np
def mask_dimensions():
    mask = np.load("../DL_masks/OFFENSE-40_richmond/Frame_300.npy")
    assert mask.shape == (224,224)
    print(mask.shape)

def h_matrix():
    M = np.load("../DL_homography_matrices/OFFENSE-40_richmond/Frame_0.npy")
    assert M.shape == (3,3)
    print(M.shape)
    print(M)
    
h_matrix()