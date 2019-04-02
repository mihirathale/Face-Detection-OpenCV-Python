import cv2
import numpy as np
import os


face_cascade = cv2.CascadeClassifier('C:/Users/MIHIR/Desktop/OpenCV/haarcascade_frontalface_default.xml')
path1 = 'C:/Users/MIHIR/Desktop/OpenCV/data/'

recognizer = cv2.face.LBPHFaceRecognizer_create()

train_imgs = []
train_ids=[]
for name in os.listdir(path1):
    for i in os.listdir(path1+name):
        img = cv2.imread(path1+name+'/'+i,cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('img',img)
        #small = cv2.resize(img, (300,300), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
        #cv2.imshow('new',small)
        image_array = np.array(img,"uint8")
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)
        for (x,y,w,h) in faces:
            roi = img[y:y+h,x:x+w]
            train_imgs.append(roi)
            train_ids.append(int(name.replace("Sub","")))
            
print(len(train_imgs))
print(len(train_ids))
    


recognizer.train(train_imgs, np.array(train_ids))
recognizer.save('trainer.xml')
