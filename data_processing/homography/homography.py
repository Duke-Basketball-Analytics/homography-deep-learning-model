import os
import cv2
import numpy as np
import json
import ipdb
from typing import List, Tuple


class HomographyHandler():
    def __init__(self, pano: np.ndarray, M1: np.ndarray, Ms: np.ndarray):
        self.M1: np.ndarray = M1
        self.Ms: np.ndarray = Ms
        self.pano: np.ndarray = pano
        self.sift = cv2.SIFT_create()
        self.pano_kp, self.pano_des = self.sift.detectAndCompute(self.pano, None)

        # FLANN parameters
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def get_keypoints(self, image: np.array) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detects keypoints and computes descriptors for the given image

        Parameters:
        -----------
        image : np.ndarray
            Image to process

        Returns:
        --------
        Tuple[List[cv2.KeyPoint], np.ndarray]
            Tuple containing list of keypoints and an array of descriptors.

        """

        kp, des = self.sift.detectAndCompute(image, None)
        return kp, des

    def get_homography(self, kp1, des1, kp2 = None, des2 = None):

        if kp2 == None:
            kp2 = self.pano_kp
            des2 = self.pano_des
        assert all(x is not None for x in (kp1, kp2, des1, des2)), "One or more variables are not defined"

        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    
    def location_transformation(self, kp:list, M:np.ndarray):
        kpt = np.array([kp[0], kp[1], 1])
        # pano_transform = M @ kpt.reshape((3, -1))
        # # print(f"Pano Transform: {pano_transform}")
        # scale_transform = self.Ms @ pano_transform
        # # print(f"Scale Transform: {scale_transform}")
        # aerial_transform = self.M1 @ scale_transform
        # # print(f"Aerial Transform: {aerial_transform}")
        kpt_trans = self.M1 @ (self.Ms @ (M @ kpt.reshape((3, -1))))
        kpt_trans = np.int32(kpt_trans / kpt_trans[-1]).ravel()
        return(kpt_trans)
    
    def evaluate_person(self, kpts:list[list], M:np.ndarray, map_dim:tuple) -> Tuple[bool, List[int]]:
        left = kpts[15]
        right = kpts[16]
        posX = int(round((left[0] + right[0]) / 2))
        posY = int(round((left[1] + right[1]) / 2))
        location = self.location_transformation([posX, posY], M = M)
        ymax, xmax = map_dim[0], map_dim[1]
        if not (0 <= location[0] <= xmax and 0 <= location[1] <= ymax):
            return False, None
        return True, location[:2]


    
def frame_cropper(frame, start_row:int, end_row:int):
    "Crop an image, but maintain the dimensions, replacing cropped parts with black pixels"
    height, width, channels = frame.shape
    frame_crop = np.zeros_like(frame)
    frame_crop[start_row:end_row, :] = frame[start_row:end_row, :]
    return frame_crop

def save_matrix(M: np.array, frame: int, video_id: str):
    # Define the directory path for saving the matrix
    directory = f"/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_homography_matrices/{video_id[:-4]}/"

    # Check if the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path for saving the matrix
    file_path = os.path.join(directory, f"Frame_{frame}.npy")

    # Save the homography matrix
    np.save(file_path, M)
    print(f"{video_id}, Frame {frame} homography matrix saved")



if __name__ == "__main__":
    # Execute from homography_deep_learning_model/ module
    from data_processing.court_masking.evaluate_person import visualize_kpts
    from utils.plotting import plt_plot

    M1 = np.load("./data_processing/homography/M1/OSU_M1.npy") # Matrix between reduced pano (for keypoint labeling) and aerial view
    Ms = np.load("./data_processing/homography/Ms/OSU_Ms.npy") # Scaling matrix between full size pano and reduced size pano
    pano = cv2.imread("./data_processing/homography/panoramics/OSU_pano.png")
    transformer = HomographyHandler(pano = pano, M1 = M1, Ms = Ms)

    video_id = "OFFENSE-40_richmond"
    video = cv2.VideoCapture(f"/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_raw/unprocessed/{video_id}.mov")
    frame_num = 0
    ok, frame = video.read()
    if not ok:
        print("Unable to read frame")
        exit()
    frame_crop = frame_cropper(frame, start_row=200, end_row=800) # Hard Coded frame cropping - need to calculate cropping based on court detection
    frame_kp, frame_des = transformer.get_keypoints(frame_crop)
    M = transformer.get_homography(kp1 = frame_kp, des1 = frame_des)
    #save_matrix(M, frame_num, video_id)

    transformed_image = cv2.warpPerspective(frame, M, (pano.shape[1], pano.shape[0]))
    # cv2.imshow('Transformed Image', transformed_image)
    # cv2.imshow('Destination Image', pano)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt_plot(transformed_image)

    f_norm = np.linalg.norm(M)
    M2 = M/f_norm
    transformed_image2 = cv2.warpPerspective(frame, M2, (pano.shape[1], pano.shape[0]))
    plt_plot(transformed_image2)

    # aerial_court = cv2.imread("homography_deep_learning_model/data_processing/homography/2d_map.png")
    # aerial_court = cv2.resize(aerial_court, (960,540))
    # mmpose = json.load(open('ohiost_44.json'))
    # mmpose_person = mmpose['frame_0000'][0][12]['keypoints']
    # left = mmpose_person[15]
    # right = mmpose_person[16]
    # posX = int(round((left[0] + right[0]) / 2))
    # posY = int(round((left[1] + right[1]) / 2))
    # location = transformer.location_transformation([posX, posY], M = M)
    # frame = visualize_kpts(image=frame, kpts=mmpose_person, person_id = 2, plot=False, ret_img=True) # Debugging: visualize individual people
    # cv2.circle(aerial_court, location[:2], 5, (0,0,255), 3)
    # # cv2.imshow('Location Mapped', aerial_court)
    # # cv2.imshow('mmpose person', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print([posX, posY])
    # print(location)
    # plt_plot(aerial_court)
    # plt_plot(frame)
    # print("Complete")


        
