import os
import threading
from io import BytesIO
from face_detection import face_detection
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

MSE_CONST = 700

class MainWindow(BoxLayout):
    face_detected = None
    detection_method = 1
    captured_image = ""

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

    def change_face_detection(self):
        if self.ids['detection_method_spinner'].text == "Haar Cascade":
            self.detection_method = 1
        else:
            self.detection_method = 2

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
        # timestr = time.strftime("%Y%m%d_%H%M%S")
        # camera.export_to_png("IMG_{}.png".format(timestr))
        self.ids['captured_img'].color = [1, 1, 1, 1]
        capturedImg = self.cam.export_as_image()

        # Flipping image vertically
        capturedImg.texture.uvpos = (0, 1)
        capturedImg.texture.uvsize = (1, -1)

        success, img_croped, self.ids['captured_img'].texture = face_detection(capturedImg.texture, self.detection_method)
        if success is True:
            self.face_detected = img_croped
            self.ids['save_descriptor_button'].disabled = False
            self.ids['compare_descriptors_button'].disabled = False
        else:
            self.ids['save_descriptor_button'].disabled = True
            self.ids['compare_descriptors_button'].disabled = True
            self.face_detected = None
            
    def save_descriptor(self):
        w = h = 128 # size of the face (128*128)
        face_resized = cv2.resize(self.face_detected, (w, h))
        timestr = time.strftime("%Y%m%d_%H%M%S")
        Generate(face_resized, file_name="my_descriptor_{}".format(timestr)).generate()

    def compare_descriptors(self):
        result = Recognition(self.face_detected).recognize()
        if result is True:
            self.ids['captured_img'].color = [0, 1, 0, 1]
            print("YES")
        else:
            self.ids['captured_img'].color = [1, 0, 0, 1]
            print("NO")

    def toggleCamera(self):
        if self.cam.play is True:
            self.cam.play = False
            self.ids['toggle_button'].text = 'Play'
        else:
            self.cam.play = True
            self.ids['toggle_button'].text = 'Pause'
            
            
class Recognition():
    def __init__(self, face_croped):
        self.face_croped = face_croped
                
    def recognize(self):
        # Normalize the face
        w = h = 128 # size of the face (128*128)
        window_size = 8 # size of the hog & lbp blocks
        face_resized = cv2.resize(self.face_croped, (w, h))
        
        # Generate descriptor
        descriptor_test = Generate(face_resized, w, h, window_size).generate()
       
        # Read descriptors
        descriptors_directory = 'descriptors/'
        if not os.path.isdir(descriptors_directory):
            os.mkdir(descriptors_directory)
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
            if(mse<MSE_CONST):
                is_authorized = True
        return is_authorized


class FaceRecoApp(App):
    def build(self):
        if not os.path.isdir('images'):
            os.mkdir('images')
        window = MainWindow()
        return window


if (__name__ == '__main__'):
    FaceRecoApp().run()
