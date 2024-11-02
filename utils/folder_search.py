import os

def list_contents(directory, param):
    # Returns list of files with appropriate file extension
    # Includes extension in returned list
    
    items = os.listdir(directory)
    if param == "movie":
        items = [item for item in items if item.endswith('.mov')]
    elif param == "folder":
        items = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    elif param == "frame":
        items = [item for item in items if item.endswith('.jpg')]
    return items