import os
import threading
from io import BytesIO

# import cv2
import time
import kivy
# import matplotlib.pyplot as plt
# import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera

kivy.require('2.0.0')

class MainWindow(BoxLayout):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Adding camera here to avoid getting error
        # The rest of the layout will be in facereco.kv
        self.cam = Camera(index=1)
        self.ids['camera'].add_widget(self.cam)

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
        # self.ids['captured_img']

        print("Captured")

    def toggleCamera(self):
        if self.cam.play is True:
            self.cam.play = False
            self.ids['toggle_button'].text = 'Play'
        else:
            self.cam.play = True
            self.ids['toggle_button'].text = 'Pause'

class FaceRecoApp(App):
    def build(self):
        if not os.path.isdir('images'):
            os.mkdir('images')
        window = MainWindow()
        return window


if (__name__ == '__main__'):
    FaceRecoApp().run()
