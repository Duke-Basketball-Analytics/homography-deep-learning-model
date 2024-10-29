import cv2
import numpy as np
from ..misc.resize_frame import resize_img

def overlay_mask(frame, court_mask, plot=True, ret_img = False):
    # Create blended mask of court
    color_mask = np.zeros_like(frame)
    court_color = [0, 255, 0]  # Green color; you can choose any RGB color
    color_mask[court_mask == 255] = court_color

    # Blend the colored mask with the original frame
    alpha = 0.3  # Transparency factor for the color mask
    beta = 1 - alpha  # Transparency factor for the original video frame
    gamma = 0  # Scalar added to each sum (usually left as 0)

    # Perform weighted addition of the two images
    overlayed_frame = cv2.addWeighted(frame, beta, color_mask, alpha, gamma)
    #overlayed_frame = resize_img(overlayed_frame)

    #vis = np.vstack((frame, cv2.resize(rectangle_frame, (frame.shape[1], frame.shape[0]))))
    if plot:
        cv2.imshow('Court Contour', overlayed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif ret_img:
        return overlayed_frame
