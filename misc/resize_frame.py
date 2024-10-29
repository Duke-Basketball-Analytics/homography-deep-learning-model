import cv2

def resize_img(img, scale = None):
    '''Resize the frame. Default is (540 x 960)'''
       # Resize the image
    if scale:
        #ipdb.set_trace()
        h, w, _ = img.shape
        w = int(w*scale)
        h = int(h*scale)
        dimensions = (w,h)
        img = cv2.resize(img, dimensions)
        
    else:
        dimensions = (960, 540)
        img = cv2.resize(img, dimensions)
    return img