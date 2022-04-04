import chainer
import chainer.links as L
import chainer.functions as F
import chainer.cuda
import argparse
import os
import cv2
import numpy as np
import glob
from network import Mynet
import config as cf
from data_loader import DataLoader
import matplotlib.pyplot as plt

model=Mynet()

#学習モデルのロード
#Loading the learning model
chainer.serializers.load_npz("./out/CNN.npz",model,strict=False)

cap=cv2.VideoCapture(0)
cascade_file='./haarcascade_frontalface_alt.xml'

while True:
    ret,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cascade=cv2.CascadeClassifier(cascade_file)

    faces=cascade.detectMultiScale(frame_gray)

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame',frame)

        face=frame[y:y+h,x:x+w]
        img=face
        img=img.astype(np.float32)
        img=cv2.resize(img,(cf.Width,cf.Height))
        img=img[:,:,(2,1,0)]
        img=img.transpose(2,0,1)
        img=img[np.newaxis,:]
        img=img/255.
        y=F.softmax(model(img).data)
        y=y.data
        print(y)
        classNum=y.argmax()
        if classNum==0:
            print("Kuramoto")
            
        elif classNum==1:
            print("Sawai")
        elif classNum==2:
            print("Sone")
        elif classNum==3:
            print("Oyama")
        elif classNum==4:
            print("横顔Kuramoto")
        elif classNum==5:
            print("Kurihara")

    key=cv2.waitKey(1) & 0xff
    if key==ord('q'):
        break
    if key==ord('s'):
        cv2.imwrite("out.jpg",frame)

out.release()
cap.release()
cv2.destroyAllWindows()
