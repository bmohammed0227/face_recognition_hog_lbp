import cv2
import numpy as np
from kivy.graphics.texture import Texture


def load_img(file_name):
    # Read the image & convert to grayscale
    img = cv2.imread(file_name)
    return img


def bgr_to_gray(img):
    # Convert the BGR image to GRAY
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def gray_to_rgb(img_gray):
    # Convert the GRAY image to RGB
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return img_rgb


def bgr_to_rgb(img):
    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def frame_to_bgr(frame):
    # Read the frame and convert it to opencv image (BGR)
    height, width = frame.height, frame.width
    img = np.frombuffer(frame.pixels, np.uint8)
    img = img.reshape(height, width, 4)
    return img


def img_to_frame(img):
    # Convert opencv image to frame texture to display it in kivy
    buffer = cv2.flip(img, 0).tostring()
    texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
    texture.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
    return texture


def save_img(img, file_name='img.png'):
    # Save opencv image to file
    cv2.imwrite(file_name, img)
    print("Image saved as : "+file_name)
