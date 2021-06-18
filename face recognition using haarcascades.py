import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

x_train = []
y_train = [] 
current_id = 0
labels_ids = {}
classifier = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
root = os.path.abspath('./images')
for root,dirs,files in os.walk(root):
    path = os.path.abspath(root)
    label = os.path.basename(root).replace(' ','-').lower()
    if files:
        for file in files:
            try:
                if file.endswith('jpg'):
                    if label in labels_ids:
                        pass
                    else:
                        labels_ids[label] = current_id
                        current_id+=1
                        label_id = labels_ids[label]
                    img_arr = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)
                    faces = classifier.detectMultiScale(img_arr,1.5,3)
                    for x,y,w,h in faces:
                        roi = img_arr[y:y+h,x:x+w]
                        x_train.append(roi)
                        y_train.append(label_id)
            except Exception as e:
                pass


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train,np.array(y_train))
reverse_ids = {v:k for (k,v) in labels_ids.items()}
capture = cv2.VideoCapture(0)
#classifier = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
while True:
    ret,frame = capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray,2.5,5)
    for x,y,w,h in faces:
        roi = gray[y:y+h,x:x+w]
        lbl,conf = recognizer.predict(roi)
        if conf>=65:
            cv2.putText(frame,reverse_ids[lbl],(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow('Face recognition Window',frame)
    if cv2.waitKey(5)&0xFF==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
        
    