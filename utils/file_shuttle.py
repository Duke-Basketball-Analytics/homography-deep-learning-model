import os
import shutil

def move(file_name, prev_path, new_path):
    # Move file to new directory
    shutil.move(prev_path, new_path)
    print(f"Moved {file_name} to {new_path}")

def is_png_file(file_path):
    # Get the file extension
    _, extension = os.path.splitext(file_path)
    # Check if the extension is .png
    return extension.lower() == '.png'