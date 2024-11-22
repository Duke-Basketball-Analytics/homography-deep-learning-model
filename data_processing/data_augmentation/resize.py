from PIL import Image
import os

def resize_image(input_path, frame_num, video_id, size):
    """
    Resizes a JPG image and saves it back as a JPG.

    Parameters:
        input_path (str): Path to the input JPG image.
        output_path (str): Path to save the resized JPG image.
        size (tuple): Desired size as (width, height).
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Ensure the image is in RGB mode (important for JPG)
            img = img.convert("RGB")
            # Resize the image
            resized_img = img.resize(size)


            # Define the directory path for saving the matrix
            output_dir = f"/Users/matth/OneDrive/Documents/DukeMIDS/DataPlus/Basketball/DL_homography/DL_frames_aug/{video_id[:-4]}/"

            # Check if the directory exists; if not, create it
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the resized image
            output_path = output_dir + f"Frame_{frame_num}.jpg"
            resized_img.save(output_path, "JPEG")

            print(f"{video_id}, Resized Frame {frame_num} saved")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    resize_image("../DL_frames/OFFENSE-40_richmond/Frame_0.jpg", 
                 "../DL_frames_aug/Frame_0.jpg", (224, 224))
