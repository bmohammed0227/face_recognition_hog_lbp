import cv2
import face_recognition
import img_tools

def detect_face(img_gray):
    # Detect the face
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    points = face_cascade.detectMultiScale(img_gray, 1.1, 15)
    img = img_tools.gray_to_rgb(img_gray)

    # Draw a rectangle around the face
    for (x, y, w, h) in points:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_croped = img_gray[y:y+h, x:x+w].copy()
            
        
        
        
    locations = face_recognition.face_locations(img_gray, model="cnn")
    y1 = locations[0][0]
    x2 = locations[0][1]
    y2 = locations[0][2]
    x1 = locations[0][3]

    # Draw a rectangle around the face
    cv2.rectangle(img_gray, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # crop image
    img_croped = img_gray[y1:y2, x1:x2].copy()
    
    # convert image to rgb
    img = img_tools.gray_to_rgb(img_gray)

    return img, img_croped



