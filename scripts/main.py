from data_processing.court_masking.court_isolation import *
from data_processing.frame_extraction.frame_extraction import extract_frames
from data_processing.homography.homography import HomographyHandler, save_matrix, frame_cropper
from utils.plotting import plt_plot
from utils.resize_frame import resize_img
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
        shutil.move(prev_path, new_path) # move video to /processed folder 
    return

def mask_processing(base_path, video_queue):
    frame_path = "/DL_frames/"
    mask_path = "/DL_masks/"
    for video_id in video_queue:
        frame_count = 0
        directory = f"{base_path}/DL_frames/{video_id[:-4]}/"
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
                save_mask(court_mask, frame_key, video_id)
            frame_count += 1

        print(f"Frames processed for video: {video_id}, {frame_count} frames.")


        
            


if __name__ == "__main__":
    # # Get the current working directory
    # current_path = os.getcwd()
    # # Print the current path
    # print(f"Current working directory: {current_path}")

    base_path = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography"
    video_queue = unprocessed_vids(base_path=base_path)
    if video_queue is not None:
        frame_processing(base_path=base_path, video_queue = video_queue)
        mask_processing(base_path=base_path, video_queue=video_queue)
    else:
        print("No videos in the video queue.")
