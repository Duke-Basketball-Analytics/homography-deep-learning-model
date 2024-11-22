import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import ipdb
from utils.plotting import plt_plot
from utils.resize_frame import resize_img
from data_processing.court_masking.overlay import overlay_mask

def isolate_court(frame, video_id, frame_num):
    '''Return a binarized mask of just the court'''
    gray_frame = binarize_erode_dilate(frame, plot=False)
    blob_frame, contours_court = blob_detection(gray_frame, plot=False)
    court_mask, contour_vertices = rectangularize_court(blob_frame.copy(), contours_court, title = f"{video_id}, Frame: {frame_num}", plot=False)
    if contour_vertices is None:
        return None
    court_mask = dilation(court_mask, iter = 5)
    #overlay_mask(frame, court_mask)
    return court_mask

def binarize_erode_dilate(img, iterations = 4, threshold=True, plot=False, save=False):
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    if threshold:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if plot:
            plt.imshow(gray, cmap='gray')
            plt.show()
        # Apply double thresholding to create a binary image within the specified range
        # Lower thresholding (all pixels above lower_thresh are set to 255)
        _, lower_result = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        # Upper thresholding (all pixels above upper_thresh are set to 0)
        _, upper_result = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Combine both thresholds to get the final mask
        img = cv2.bitwise_and(lower_result, upper_result)
    # plt.imshow(img_otsu, cmap='gray')
    # plt.show()

    # kernel = np.array([[0, 0, 0],
    #                    [1, 1, 1],
    #                    [0, 0, 0]], np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    img_otsu = cv2.erode(img, kernel, iterations=iterations)
    if plot:
        plt.imshow(img_otsu, cmap='gray')
        plt.show()
    img_otsu = cv2.dilate(img_otsu, kernel, iterations=iterations*2)
    if plot:
        plt.imshow(img_otsu, cmap='gray')
        plt.show()
    return img_otsu

def blob_detection(frame, plot=False):
     # BLOB FILTERING & BLOB DETECTION

    # adding a little frame to enable detection
    # of blobs that touch the borders
    frame[-4: -1] = frame[0:3] = 0
    frame[:, 0:3] = frame[:, -4:-1] = 0

    mask = np.zeros(frame.shape, dtype=np.uint8)
    cnts = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_court = []

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #threshold_area = 5000
    threshold_area = 20000 # THIS IS ONLY CHNGED BECAUSE THE FRAME IS HIGHER RESOLUTION
    for c in cnts:
        area = cv2.contourArea(c)
        if area > threshold_area:
            cv2.drawContours(mask, [c], -1, 255, -1)
            contours_court.append(c)
    # ipdb.set_trace()
    frame = mask
    if plot: plt_plot(frame, title="After Blob Detection", cmap="gray",
                      save_path=None)
    
    return frame, contours_court

def rectangularize_court(frame, contours_court, title, plot=False, convex_hull=False):

    simple_court = np.zeros(frame.shape)
    if len(contours_court) > 1:
        final_contours = np.vstack(contours_court)
    elif len(contours_court) == 1:
        final_contours = contours_court[0]
    else:
        print("No court detected.")
        return simple_court, None
    # convex hull
    hull = cv2.convexHull(final_contours)
    cv2.drawContours(frame, [hull], 0, 100, 2)
    if plot: 
        fig = plt_plot(frame, title="After ConvexHull", cmap="gray",
                      additional_points=hull.reshape((-1, 2)),
                      save_path=None,
                      send_back=True)
        if convex_hull:
            return fig, None
        
    # ipdb.set_trace()
    # fitting a poly to the hull
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    corners = approx.reshape(-1, 2)
    cv2.drawContours(frame, [approx], 0, 100, 5)
    cv2.drawContours(simple_court, [approx], 0, 255, 3)

    if plot:
        plt_plot(frame, title="After Rectangular Fitting", cmap="gray", additional_points=hull.reshape((-1, 2)),
                 save_path=None)
        plt_plot(simple_court, title=f"Rectangularized Court {title}", cmap="gray", additional_points=hull.reshape((-1, 2)),
                 save_path=None)
        print("simplified contour has", len(approx), "points")
    

    isolated_court = np.zeros(simple_court.shape, dtype=np.uint8)
    cv2.fillPoly(isolated_court, [approx], 255)
    return isolated_court, approx

def dilation(img, k = 3, iter = 1):
    # Define the kernel size for dilation
    kernel_size = k
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate the mask
    mask = img
    dilated_mask = cv2.dilate(mask, kernel, iterations=iter)

    return dilated_mask

def save_mask(mask: np.array, frame: int, video_id: str):
    # Define the directory path for saving the matrix
    directory = f"/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_masks/{video_id[:-4]}/"

    # Check if the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path for saving the matrix
    file_path = os.path.join(directory, f"Frame_{frame}.npy")

    # Save the homography matrix
    np.save(file_path, mask)
    print(f"{video_id}, Frame {frame} court mask saved")

if __name__ == "__main__":
    video_id = "OFFENSE-40_richmond.mov"
    frame_key = 0
    video_path = f"/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_raw/{video_id}"
    # cap = cv2.VideoCapture(video_path)
    # ret, frame = cap.read()

    frame_path = "../DL_frames_aug/OFFENSE-40_richmond/Frame_0.jpg"
    frame = cv2.imread(frame_path)
    court_mask = isolate_court(frame, video_id, frame_key)
    # save_mask(court_mask, frame_key, video_id)
    plt_plot(court_mask)
    