# importing necessary libraries
import os
import numpy as np
import cv2

# generating data with labels
x_train = []
y_train = []
current_id = 0
label_ids = {} # dict which contain all labels with numeric digit associated with it

classifier = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')# Haarcascades classifier which detect face 
main_dir = 'D:\computer vision\images'
for root,subdir,files in os.walk(main_dir):
    path = os.path.abspath(root)
    label = os.path.basename(root).replace(' ','-').title()
    for file in files:
        if label in label_ids:
            pass
        elif label not in label_ids:
            label_ids[label] = current_id
            current_id+=1
            label_id = label_ids[label]
        img_arr = cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE) # reading image converting to array
        faces = classifier.detectMultiScale(img_arr,1.4,4) # detecting face
        for x,y,w,h in faces:
            roi = img_arr[y:y+h,x:x+w]         # image region of interest i.e. face
            x_train.append(roi)
            y_train.append(label_id)
            
recognizer = cv2.face.LBPHFaceRecognizer_create()        # LBPH Face recognizer
recognizer.train(x_train,np.array(y_train))              # training
reverse_ids = {v:k for (k,v) in label_ids.items()}
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()                             # taking picture using web camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray,1.4,4)
    for x,y,w,h in faces:
        roi = gray[y:y+h,x:x+w]
        lbl,conf = recognizer.predict(roi)           # predicting the face
        if conf>=75:
            cv2.putText(frame,reverse_ids[lbl],(x-10,y-20),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5) # identify what detected
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow('Face recognition Window',frame)
    if cv2.waitKey(5)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()