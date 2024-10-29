from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def upload_file_to_drive(file_path, folder_id):
    gauth = GoogleAuth()

    # Load the credentials from the settings.yaml file
    gauth.LoadCredentialsFile("settings.yaml")

    # If no valid credentials are available, go through the authentication process
    if gauth.credentials is None or gauth.access_token_expired:
        # Automatically refreshes the token if it's expired
        if gauth.credentials:
            gauth.Refresh()  # Refresh the token if available
        else:
            gauth.LocalWebserverAuth()  # Creates a local web server for authentication
        gauth.SaveCredentialsFile("settings.yaml")  # Save the new credentials

    drive = GoogleDrive(gauth)

    # Create a file instance for the file you want to upload
    file_to_upload = drive.CreateFile({'title': os.path.basename(file_path), 'parents': [{'id': folder_id}]})
    file_to_upload.SetContentFile(file_path)
    file_to_upload.Upload()  # Upload the file

    print(f"Uploaded file: {file_to_upload['title']} with ID: {file_to_upload['id']}")
    
if __name__ == "__main__":
    # Set the file path and folder ID
    file_path = '../2d_map.png'  # Replace with your file path
    folder_id = '1isFq36BjzGrfpKj-49bAEbZDMd8SH5CM'  # Replace with your Google Drive folder ID
    
    upload_file_to_drive(file_path, folder_id)
