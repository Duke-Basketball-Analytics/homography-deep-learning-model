import cv2
import os

def extract_frames(video_id, skip_frames=100):
    # Find game footage
    video_path = f"DL_raw/{video_id}.mov"
    if not os.path.exists(video_path):
        print("VIDEO NOT FOUND")
        return None
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Define the directory path for saving the matrix
    directory = f"DL_frames/{video_id}/"
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
    video_id = "OFFENSE-40_richmond"
    # video_path = f"/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_raw/{video_id}.mov"
    # output_folder = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_frames"
    extract_frames(video_id, skip_frames=100)