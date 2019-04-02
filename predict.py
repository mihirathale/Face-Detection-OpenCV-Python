import cv2
import numpy as np
import os


face_cascade = cv2.CascadeClassifier('C:/Users/MIHIR/Desktop/OpenCV/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.xml')

people = ["","Tyrrian","Jaime","Loki","Mihir"]

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        ret_id, acc = recognizer.predict(roi_gray)
        label_txt = people[ret_id]
        if (acc > 50 and acc <75):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,label_txt,(x,y), font, 1, (0,255,100), 2, cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'unknown',(x,y), font, 1, (0,255,100), 2, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

        
    
    
