import cv2
import os

def extract_frames(video_path, output_folder, skip_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Frame {frame_count} saved")
        frame_count += 1
    cap.release()

if __name__ == "__main__":
    video_path = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_raw/OFFENSE-40_richmond.mov"
    output_folder = "/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_frames"
    extract_frames(video_path, output_folder)