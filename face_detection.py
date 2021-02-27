import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import atan, pi
import pickle
from kivy.graphics.texture import Texture

import img_tools

def detect_face(img_gray):
    # Detect the face
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    points = face_cascade.detectMultiScale(img_gray, 1.1, 15)
    img = img_tools.gray_to_rgb(img_gray)

    # Draw a rectangle around the face
    for (x, y, w, h) in points:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img
