from data_processing.court_masking.court_isolation import *
from data_processing.frame_extraction.frame_extraction import extract_frames
from data_processing.homography.homography import HomographyHandler, save_matrix, frame_cropper
from utils.plotting import plt_plot
from utils.resize_frame import resize_img
from google_drive.data_upload import upload_file_to_drive
import os

def processing_pipeline(video_id):
    extract_frames(video_id, skip_frames=100)
    return


if __name__ == "__main__":
    # # Get the current working directory
    # current_path = os.getcwd()
    
    # # Print the current path
    # print(f"Current working directory: {current_path}")
    processing_pipeline(video_id = "OFFENSE-40_richmond")