

from data_processing.court_masking.court_isolation import *
from data_processing.frame_extraction.frame_extraction import extract_frames
from data_processing.homography.homography import HomographyHandler, save_matrix, frame_cropper
from data_processing.data_augmentation.resize import resize_image, resize_binary_mask
from utils.plotting import plt_plot
from utils.folder_search import list_contents
from google_drive.data_upload import upload_file_to_drive
from utils.file_shuttle import is_png_file
import os
import shutil

def unprocessed_vids(base_path):
    unprocessed_dir = "/DL_raw/unprocessed/"
    unprocessed_vids = list_contents(directory = base_path + unprocessed_dir,
                                    param = "movie")
    if len(unprocessed_vids) > 0:
        return unprocessed_vids
    else: return None

def frame_processing(base_path, video_queue):
    unprocessed_dir = "/DL_raw/unprocessed/"
    processed_dir= "/DL_raw/processed/"

    for video_id in video_queue:
        extract_frames(base_path = base_path, unprocessed_dir = unprocessed_dir, video_id = video_id, skip_frames=100)
        prev_path = base_path + unprocessed_dir + video_id
        new_path = base_path + processed_dir + video_id
        #shutil.move(prev_path, new_path) # move video to /processed folder 
    
    return

def augmentation_processing(base_path, video_queue):
    frame_path = "/DL_frames/"
    mask_path = "/DL_masks/"
    for video_id in video_queue:
        frame_count = 0
        directory = f"{base_path}{frame_path}{video_id[:-4]}/"
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            continue
        frames = list_contents(directory, param='frame')
        if len(frames) == 0:
            print(f"Directory Exists - No valid frames (.jpg) found in directory")
            continue
        for frame_name in frames:
            frame_path = directory + frame_name
            frame_key = frame_name.split('_')[1].split('.')[0]
            resize_image(input_path=frame_path, frame_num = frame_key, video_id=video_id, size=(224, 224))
            frame_count += 1

        print(f"Frames processed for video: {video_id}, {frame_count} frames.")


def mask_processing(base_path, video_queue):
    frame_path = "/DL_frames/"
    mask_path = "/DL_masks/"
    for video_id in video_queue:
        frame_count = 0
        directory = f"{base_path}{frame_path}{video_id[:-4]}/"
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            continue
        frames = list_contents(directory, param='frame')
        if len(frames) == 0:
            print(f"Directory Exists - No valid frames (.jpg) found in directory")
            continue
        for frame_name in frames:
            frame_path = directory + frame_name
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame_key = frame_name.split('_')[1].split('.')[0]
                court_mask = isolate_court(frame, video_id, frame_key)
                if court_mask is None:
                    print(f"Court Mask Not Saved {video_id} Frame {frame_key}")
                    continue
                resized_mask = resize_binary_mask(court_mask, size=(224,224))    
                save_mask(resized_mask, frame_key, video_id)
            frame_count += 1

        print(f"Frames processed for video: {video_id}, {frame_count} frames.")

def homography_processing(base_path, video_queue):
    frame_path = "/DL_frames/"
    homography_path = "/DL_homography_matrices/"
    pano = cv2.imread("data_processing/homography/panoramics/OSU_pano.png")
    Ms = np.load("data_processing/homography/Ms/OSU_Ms.npy")
    M1 = np.load("data_processing/homography/M1/OSU_M1.npy")
    for video_id in video_queue:
        frame_count = 0
        directory = f"{base_path}{frame_path}{video_id[:-4]}/" # directory where individual frames are saved
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            continue
        frames = list_contents(directory, param='frame') # only retrieve frames (images)
        if len(frames) == 0:
            print(f"Directory Exists - No valid frames (.jpg) found in directory")
            continue
        for frame_name in frames: # iterated through each frame
            frame_path = directory + frame_name
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame_key = frame_name.split('_')[1].split('.')[0]
                transformer = HomographyHandler(pano = pano, M1 = M1, Ms = Ms) #Object for handling homography calculations
                frame_crop = frame_cropper(frame, start_row=200, end_row=800) #Manual cropping to improve SIFT KP matching - update later with court mask
                frame_kp, frame_des = transformer.get_keypoints(frame_crop) #SIFT keypoints and descriptors
                M = transformer.get_homography(kp1 = frame_kp, des1 = frame_des) # Calculate matrix
                f_norm = np.linalg.norm(M) # Calculate Frobenius Norm
                M_norm = M/f_norm # Normalize matrix for improved deep learning model
                save_matrix(M_norm, frame_key, video_id)
            frame_count += 1
        print(f"Homography Matrices processed for video: {video_id}, {frame_count} frames.")
        
            


if __name__ == "__main__":
    # Run from parent package homography_deep_learning_model/
    # python -m scripts.main

    # # Get the current working directory
    # current_path = os.getcwd()
    # # Print the current path
    # print(f"Current working directory: {current_path}")

    base_path = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography"
    video_queue = unprocessed_vids(base_path=base_path)
    if video_queue is not None:
        frame_processing(base_path=base_path, video_queue = video_queue)
        augmentation_processing(base_path=base_path, video_queue = video_queue)
        mask_processing(base_path=base_path, video_queue=video_queue)
        homography_processing(base_path=base_path, video_queue=video_queue)
    else:
        print("No videos in the video queue.")
