import numpy as np
import cv2
import matplotlib.pyplot as plt
import ipdb
from ...utils.plotting import plt_plot

'''This file contains all the functions required to evaluate whether a player is standing on the court.'''

def person_validity(mask, kpts):
    '''Using the mask and the location of the person's ankles, determine if the player is on the court.
       If the player is on the court, return True, if the player isn't on the court return False.'''
    left = kpts[15]
    right = kpts[16]
    posX = int(round((left[0] + right[0]) / 2))
    posY = int(round((left[1] + right[1]) / 2))
    if posY >= mask.shape[0]: posY = mask.shape[0]-1
    if posX >= mask.shape[1]: posX = mask.shape[1]-1
    if mask[posY, posX] == 255:
        return True
    else: 
        return False

def convert_to_integers(coordinates):
    return [(int(round(x)), int(round(y))) for x, y in coordinates]

def visualize_kpts(image, kpts, person_id, plot=True, ret_img=False):
    '''Visualize the keypoints from mmpose on the image. Debugging tool.'''
    # Read the image
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Draw keypoints on the image
    for keypoint in kpts:
        x, y = keypoint
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Display the image with keypoints
    if plot:
        # plt.figure(figsize=(10, 10))
        # plt.title(f"Person Index: {person_id}")
        # plt.imshow(image)
        # plt.axis('on')
        # plt.show()
        plt_plot(image)
    elif ret_img:
        return image

def visualize_center(image, kpts, person_id, plot=True, ret_img=False):

    locs = [5,6,11,12]
    x_cent, y_cent = find_center(kpts, locs)
    buff = 10
    upper_left = (x_cent - 2*buff, y_cent - 2*buff)
    bottom_right = (x_cent + 2*buff, y_cent + 2*buff)
    cv2.circle(image, (x_cent, y_cent),5, (0,255,0), -1)
    cv2.rectangle(image, upper_left, bottom_right, (0,255,0), 3)
    cv2.putText(image, f'MMPose Detections', (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    if plot:
        plt_plot(image)
    if ret_img:
        return image
 
def find_center(kpts, locs = None):
    if locs is None:
        locs = [x for x in range(len(kpts))]
    y_cent = int(round(np.average([kpts[x][1] for x in locs])))
    x_cent = int(round(np.average([kpts[x][0] for x in locs])))
    return x_cent, y_cent

if __name__ == "__main__":
    pass
    
    
