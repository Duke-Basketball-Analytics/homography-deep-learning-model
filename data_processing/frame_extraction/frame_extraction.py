import cv2
import os

def extract_frames(base_path, unprocessed_dir, video_id, skip_frames=100):
    # Find game footage
    # This should be changed to google drive or a better pipeline location, but for now the scripts are executed from the 
    # homography_deep_learning_model module so I can't access the pipeline directories through relative paths
    video_path = base_path + unprocessed_dir + video_id
    if not os.path.exists(video_path):
        print("VIDEO NOT FOUND")
        print(f"Video Path:\n{video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Define the directory path for saving the matrix
    directory = f"{base_path}/DL_frames/{video_id[:-4]}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            frame_filename = os.path.join(directory, f'Frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Frame {frame_count} saved")
        frame_count += 1
    cap.release()

if __name__ == "__main__":
    video_id = "OFFENSE-40_richmond.mov"
    # video_path = f"/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_raw/{video_id}.mov"
    # output_folder = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_frames"
    base_path = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography"
    extract_frames(base_path, video_id, skip_frames=100)