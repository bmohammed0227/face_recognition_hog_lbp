import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import atan, pi
import pickle
from kivy.graphics.texture import Texture

import img_tools


points = np.ndarray([])
img_croped = None

def detect_face(img_gray):
    # Detect the face
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    global points
    points = face_cascade.detectMultiScale(img_gray, 1.1, 15)
    img = img_tools.gray_to_rgb(img_gray)
    global img_croped
    img_croped = img

    # Draw a rectangle around the face
    for (x, y, w, h) in points:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_croped = img_gray[y:y+h, x:x+w].copy()
            
    return img, img_croped


def normalize_face(img_gray):
    # normalize the face
    global img_croped
    if len(points) != 0:
        for (x, y, w, h) in points:
            img_croped = img_gray[y:y+h, x:x+w].copy()
        img_size = 128  # size of the face
        img_resized = cv2.resize(img_croped, (img_size, img_size))
        cv2.imwrite("norm_img.jpg", img_resized)
        print("normalized image generated")
