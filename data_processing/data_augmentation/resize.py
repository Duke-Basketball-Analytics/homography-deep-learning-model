from PIL import Image
import os
from utils.plotting import plt_plot

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

import numpy as np
from PIL import Image

def resize_binary_mask(binary_mask, size):
    """
    Resizes a binary mask represented as a numpy array to the desired size.

    Parameters:
        binary_mask (np.ndarray): Input binary mask with shape (height, width).
        size (tuple): Desired size as (width, height).

    Returns:
        np.ndarray: Resized binary mask as a numpy array with values 0 and 1.
    """
    try:
        # Convert the numpy array to a PIL Image
        pil_image = Image.fromarray(binary_mask.astype(np.uint8))
        # Resize the image
        resized_image = pil_image.resize(size, resample=Image.NEAREST)
        # Convert back to numpy array and binarize (ensure 0 and 1 only)
        resized_binary_mask = np.array(resized_image) // 255

        return resized_binary_mask.astype(np.uint8)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    # resize_image("../DL_frames/OFFENSE-40_richmond/Frame_0.jpg", 
    #              "../DL_frames_aug/Frame_0.jpg", (224, 224))
    mask = np.load("../DL_masks/OFFENSE-40_richmond/Frame_300.npy")
    print(mask.shape)
    resized = resize_binary_mask(mask, size = (224,224))
    print(resized.shape)
    plt_plot(mask)
    plt_plot(resized)
