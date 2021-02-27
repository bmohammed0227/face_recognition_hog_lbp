import cv2
import numpy as np
from math import atan, pi

class Generate :
    def __init__(self, face_resized, w, h, window_size):
        self.face_resized = face_resized
        self.w = w
        self.h = h
        self.window_size = window_size
        pass
        
    
    def generate(self):
        face_resized = self.face_resized
        w = self.w 
        h = self.h 
        window_size = self.window_size
        
         # ----LBP----

        # duplicate edges
        face_resized = cv2.vconcat([face_resized, face_resized[h-1:h, 0:w]]) # +1 row at the end
        face_resized = cv2.vconcat([face_resized[0:1, 0:w], face_resized]) # +1 row at the start
        face_resized = cv2.hconcat([face_resized, face_resized[0:h+2, w-1:w]]) # +1 column at the end 
        face_resized = cv2.hconcat([face_resized[0:h+2, 0:1], face_resized]) # +1 column at the start 
        
        # calculate lbp
        img_lbp = np.zeros((h, w),np.uint8) 
        for y in range(-1, h-1):
            for x in range(-1, w-1):
                mini_matrix =  face_resized[(y)+1:(y+3)+1, (x)+1:(x+3)+1].copy()
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
        for y in range(0, h, window_size):
            for x in range(0, w, window_size):
                block_lbp = img_lbp[y:y+window_size, x:x+window_size].copy()
                list_hist_lbp.append(np.histogram( block_lbp,256,[0,256])[0])
        
        # ----HOG----
        
        #calculate gx and gy and teta 
        Gx = np.zeros((h, w),np.int8) 
        Gy = np.zeros((h, w),np.int8) 
        teta = np.zeros((h, w),np.int8) 
        for y in range(-1, h-1):
            for x in range(-1, w-1):
                mini_matrix =  face_resized[(y)+1:(y+3)+1, (x)+1:(x+3)+1].copy()
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
                    if (gx<0):
                        degree_val_corrected += 180
                    if (gx>0 and gy<0):
                        degree_val_corrected += 360
                    if(337.5<=degree_val_corrected or degree_val_corrected<22.5):
                        teta[y+1][x+1] = 0
                    if(22.5<=degree_val_corrected<67.5):
                        teta[y+1][x+1] = 1
                    if(67.5<=degree_val_corrected<112.5):
                        teta[y+1][x+1] = 2
                    if(112.5<=degree_val_corrected<157.5):
                        teta[y+1][x+1] = 3
                    if(157.5<=degree_val_corrected<202.5):
                        teta[y+1][x+1] = 4
                    if(202.5<=degree_val_corrected<247.5):
                        teta[y+1][x+1] = 5
                    if(247.5<=degree_val_corrected<292.5):
                        teta[y+1][x+1] = 6
                    if(292.5<=degree_val_corrected<337.5):
                        teta[y+1][x+1] = 7
        # calculate hog histograms
        list_hist_hog = []
        for y in range(0, h, window_size):
            for x in range(0, w, window_size):
                block_hog = teta[y:y+window_size, x:x+window_size].copy()
                list_hist_hog.append(np.histogram( block_hog,256,[0,256])[0])
        
        # concatenate hog & lbp
        descriptor_test = []
        for i in range(0, int((w/window_size)**2)):
            descriptor_test.append((list_hist_lbp[i], list_hist_hog[i]))
            
        return descriptor_test
        
                