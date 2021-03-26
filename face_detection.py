import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import atan, pi
import pickle
from kivy.graphics.texture import Texture
import face_recognition

import img_tools


points = np.ndarray([])
img_croped = None
img_resized = None
img_lbp = None
window_size = None
width = None
height = None
list_hist_lbp = []
list_hist_hog = []
img_size = 0
success = False


def detect_face(img_gray):
    # Detect the face
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    global points, img_croped, success
    points = face_cascade.detectMultiScale(img_gray, 1.1, 15)
    img = img_tools.gray_to_rgb(img_gray)
    img_croped = img

    # Draw a rectangle around the face
    for (x, y, w, h) in points:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_croped = img_gray[y:y+h, x:x+w].copy()
            
    if len(points) != 0:
        success = True
    else:
        success = False

    return img, img_croped


def detect_face2(img_gray):
    # Detect the face
    locations = face_recognition.face_locations(img_gray, model="cnn")

    global img_croped, success

    # convert image to rgb
    img = img_tools.gray_to_rgb(img_gray)

    img_croped = img

    print(locations)
    # Draw a rectangle around the face
    for (y1, x2, y2, x1) in locations:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # crop image
        img_croped = img_gray[y1:y2, x1:x2].copy()

    if len(locations) != 0:
        success = True
    else:
        success = False

    return img, img_croped

def normalize_face(img_gray):
    # normalize the face
    global img_croped, img_resized, img_size
    if len(points) != 0:
        for (x, y, w, h) in points:
            img_croped = img_gray[y:y+h, x:x+w].copy()
        img_size = 128  # size of the face
        img_resized = cv2.resize(img_croped, (img_size, img_size))
        return img_resized


def lbp_face():
    global window_size, height, width, img_lbp, img_resized, list_hist_lbp
    if img_resized is not None:
        window_size = 8  # size of the hog & lbp blocks
        width = img_resized.shape[1]
        height = img_resized.shape[0]

        # ----LBP----

        # duplicate edges
        # +1 row at the end
        img_resized = cv2.vconcat([img_resized,
                                   img_resized[height-1:height, 0:width]])
        # +1 row at the start
        img_resized = cv2.vconcat([img_resized[0:1, 0:width],
                                   img_resized])
        # +1 column at the end
        img_resized = cv2.hconcat([img_resized,
                                   img_resized[0:height+2, width-1:width]])
        # +1 column at the start
        img_resized = cv2.hconcat([img_resized[0:height+2, 0:1],
                                   img_resized])

        # # calculate lbp
        img_lbp = np.zeros((height, width), np.uint8)
        for y in range(-1, height-1):
            for x in range(-1, width-1):
                mini_matrix = img_resized[(y)+1:(y+3)+1, (x)+1:(x+3)+1].copy()
                seuil = mini_matrix[1][1]
                i1 = 1 if mini_matrix[0][0] >= seuil else 0
                i2 = 2 if mini_matrix[0][1] >= seuil else 0
                i3 = 4 if mini_matrix[0][2] >= seuil else 0
                i4 = 8 if mini_matrix[1][2] >= seuil else 0
                i5 = 16 if mini_matrix[2][2] >= seuil else 0
                i6 = 32 if mini_matrix[2][1] >= seuil else 0
                i7 = 64 if mini_matrix[2][0] >= seuil else 0
                i8 = 128 if mini_matrix[1][1] >= seuil else 0
                somme = i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8
                img_lbp[y+1][x+1] = somme

        # calculate lbp histograms
        list_hist_lbp = []
        for y in range(0, height, window_size):
            for x in range(0, width, window_size):
                block_lbp = img_lbp[y:y+window_size, x:x+window_size].copy()
                list_hist_lbp.append(np.histogram(block_lbp, 256, [0, 256])[0])


def hog_face():
    global height, width, window_size, img_resized, list_hist_hog
    if img_resized is not None:
        # ----HOG----

        #calculate gx and gy and teta 
        Gx = np.zeros((height, width), np.int8)
        Gy = np.zeros((height, width), np.int8)
        teta = np.zeros((height, width), np.int8)
        for y in range(-1, height-1):
            for x in range(-1, width-1):
                mini_matrix = img_resized[(y)+1:(y+3)+1, (x)+1:(x+3)+1].copy()
                Gx[y+1][x+1] = mini_matrix[1][0] - mini_matrix[1][2]
                Gy[y+1][x+1] = mini_matrix[0][1] - mini_matrix[2][1]
                gx = Gx[y+1][x+1]
                gy = Gy[y+1][x+1]
                if gx == 0:
                    if gy == 0:
                        teta[y+1][x+1] = -1
                    else :
                        if gy> 0:
                            teta[y+1][x+1] = 2
                        else :
                            teta[y+1][x+1] = 6
                else :
                    val = atan(gy/gx)
                    degree_val = atan(val)*(180/pi)
                    degree_val_corrected = degree_val
                    if (gx < 0):
                        degree_val_corrected += 180
                    if (gx > 0 and gy < 0):
                        degree_val_corrected += 360
                    if(337.5 <= degree_val_corrected or degree_val_corrected < 22.5):
                        teta[y+1][x+1] = 0
                    if(22.5 <= degree_val_corrected < 67.5):
                        teta[y+1][x+1] = 1
                    if(67.5 <= degree_val_corrected < 112.5):
                        teta[y+1][x+1] = 2
                    if(112.5 <= degree_val_corrected < 157.5):
                        teta[y+1][x+1] = 3
                    if(157.5 <= degree_val_corrected < 202.5):
                        teta[y+1][x+1] = 4
                    if(202.5 <= degree_val_corrected < 247.5):
                        teta[y+1][x+1] = 5
                    if(247.5 <= degree_val_corrected < 292.5):
                        teta[y+1][x+1] = 6
                    if(292.5 <= degree_val_corrected < 337.5):
                        teta[y+1][x+1] = 7

        # calculate hog histograms
        list_hist_hog = []
        for y in range(0, height, window_size):
            for x in range(0, width, window_size):
                block_hog = teta[y:y+window_size, x:x+window_size].copy()
                list_hist_hog.append(np.histogram(block_hog, 256, [0, 256])[0])




def face_detection(frame, detection_method):
    img = img_tools.frame_to_bgr(frame)
    img_gray = img_tools.bgr_to_gray(img)
    if detection_method == 1:
        img_result, img_croped = detect_face(img_gray)
    elif detection_method == 2:
        img_result, img_croped = detect_face2(img_gray)
    # normalized_img = normalize_face(img_gray)
    # lbp_img = lbp_face()
    # hog_face()
    # serialize_descriptor()
    return success, img_croped, img_tools.img_to_frame(img_result)
