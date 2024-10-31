from data_processing.court_masking.court_isolation import *
from data_processing.frame_extraction.frame_extraction import extract_frames
from data_processing.homography.homography import HomographyHandler, save_matrix, frame_cropper
from utils.plotting import plt_plot
from utils.resize_frame import resize_img
from utils.folder_search import list_folders
from utils.file_shuttle import move
from google_drive.data_upload import upload_file_to_drive
import os


def frame_processing(full_path, video_ids:list):
    unprocessed_path = full_path + "/DL_raw/unprocessed"
    processed_path = full_path + "/DL_raw/processed"
    unprocessed_vids = list_folders(directory = unprocessed_path,
                                    param = "movie")
    for video_id in unprocessed_vids:
        extract_frames(full_path = full_path, video_id = video_id[:-4], skip_frames=100)
        move(file_name=video_id, prev_path=unprocessed_path, new_path=processed_path)
    
    return


if __name__ == "__main__":
    # # Get the current working directory
    # current_path = os.getcwd()
    # # Print the current path
    # print(f"Current working directory: {current_path}")

    full_path = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography"
    frame_processing(full_path = full_path, video_ids = ["OFFENSE-40_richmond"])