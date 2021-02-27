import os
import threading
from io import BytesIO
import face_detection, img_tools
from generate_descriptor import Generate
import pickle
from os import walk
import cv2
import time
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera


kivy.require('2.0.0')

class MainWindow(BoxLayout):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Adding camera here to avoid getting error
        # The rest of the layout will be in facereco.kv
        self.cam = None
        try:
            self.cam = Camera(index=0)
            self.ids['camera'].add_widget(self.cam)
        except:
            pass

    def reload_camera(self):
        index = 0
        index_text = self.ids['index_spinner'].text
        if index_text.endswith("1"):
            index = 1
        elif index_text.endswith("2"):
            index = 2
        try:
            if self.cam is None:
                self.cam = Camera(index=index)
                self.ids['camera'].add_widget(self.cam)
            else:
                self.cam.index = index
        except:
            if self.cam is not None:
                self.ids['camera'].remove_widget(self.cam)
                self.cam = None
        
    def capture(self):
        camera = self.cam
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        capturedImg = self.cam.export_as_image()
        self.ids['captured_img'].texture = capturedImg.texture

        # Flipping image vertically
        capturedImg.texture.uvpos = (0, 1)
        capturedImg.texture.uvsize = (1, -1)

        self.ids['captured_img'].texture = capturedImg.texture
        img = img_tools.frame_to_bgr(capturedImg.texture)
        img_gray = img_tools.bgr_to_gray(img)
        img_result, img_croped = face_detection.detect_face(img_gray)
        self.ids['captured_img'].texture = img_tools.img_to_frame(img_result)
        # self.ids['captured_img']

        print("Captured")
        Recognition(img_croped)

    def toggleCamera(self):
        if self.cam.play is True:
            self.cam.play = False
            self.ids['toggle_button'].text = 'Play'
        else:
            self.cam.play = True
            self.ids['toggle_button'].text = 'Pause'
            
            
class Recognition():
    def __init__(self, face_croped):
                
        # Normalize the face
        w = h = 128 # size of the face (128*128)
        window_size = 8 # size of the hog & lbp blocks
        face_resized = cv2.resize(face_croped, (w, h))
        
        # Generate descriptor
        descriptor_test = Generate(face_resized, w, h, window_size).generate()
       
        # Read descriptors
        descriptors_directory = 'descriptors/'
        _, _, descriptors_names = next(walk(descriptors_directory))
        descriptors = []
        for descriptor in descriptors_names:
            with open(descriptors_directory+descriptor, 'rb') as file:
                descriptors.append(pickle.load(file))
        
        is_authorized = False
        for i in range(len(descriptors)):
            # Compare the two descriptors
            descriptor = descriptors[i]
            mse_lbp = []
            mse_hog = []
            mse = 0;
            for j in range(len(descriptor)):
                mse_lbp.append(((descriptor[j][0]-descriptor_test[j][0])**2).mean())
                mse_hog.append(((descriptor[j][1]-descriptor_test[j][1])**2).mean())
                mse += mse_lbp[j] + mse_hog[j]
            print(mse)
            if(mse<700):
                is_authorized = True
        if is_authorized:
            print("YES")
        else:
            print("NO")


class FaceRecoApp(App):
    def build(self):
        if not os.path.isdir('images'):
            os.mkdir('images')
        window = MainWindow()
        return window


if (__name__ == '__main__'):
    FaceRecoApp().run()
