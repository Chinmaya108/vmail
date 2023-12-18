import cv2
import numpy as np
#import urllib.request
#import urllib.parse
import random
import os
import subprocess
import webbrowser
import gtts
from playsound import playsound  

face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.yml")
ide=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL

while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        ide,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf < 60):
            if(ide==14):
                ide="Chinmaya"
                print('welcome')
                  
                t1 = gtts.gTTS("DETECTED SUCCESSFULLY")
                t1.save("A.mp3")
                playsound("A.mp3")
                cv2.waitKey(5)
                cam.release()
                cv2.destroyAllWindows()
                webbrowser.open_new_tab('http://127.0.0.1:8000/')
                os.system('python manage.py runserver')


            elif(ide==15):
                ide="Darshan"
            elif(ide==50):
                ide="Shrivathsa"
            elif(ide==13):
                ide="Chethan"
        else:
            ide="Unknown"
            t1 = gtts.gTTS("Not Detected")
            t1.save("b.mp3")
            playsound("b.mp3")
        cv2.putText(img,str(ide),(x,y+h),font,2,255,2)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
