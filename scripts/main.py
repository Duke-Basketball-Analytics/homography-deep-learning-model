from data_processing.court_masking.court_isolation import *
from data_processing.frame_extraction.frame_extraction import extract_frames
from data_processing.homography.homography import HomographyHandler, save_matrix, frame_cropper
from utils.plotting import plt_plot
from utils.resize_frame import resize_img
from utils.folder_search import list_folders
from utils.file_shuttle import move
from google_drive.data_upload import upload_file_to_drive
import os
import shutil


def frame_processing(base_path):
    unprocessed_dir = "/DL_raw/unprocessed/"
    processed_dir= "/DL_raw/processed/"
    unprocessed_vids = list_folders(directory = base_path + unprocessed_dir,
                                    param = "movie")
    for video_id in unprocessed_vids:
        extract_frames(base_path = base_path, unprocessed_dir = unprocessed_dir, video_id = video_id, skip_frames=100)
        prev_path = base_path + unprocessed_dir + video_id
        new_path = base_path + processed_dir + video_id
        shutil.move(prev_path, new_path) # move video to /processed folder 
    
    return


if __name__ == "__main__":
    # # Get the current working directory
    # current_path = os.getcwd()
    # # Print the current path
    # print(f"Current working directory: {current_path}")

    base_path = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography"
    frame_processing(base_path=base_path)