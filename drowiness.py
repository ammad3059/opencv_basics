import cv2
import dlib
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    ratio = A+B/(2.0*C)
    return ratio

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
capture = cv2.VideoCapture(0)
while True:
    ret,frame = capture.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(frame)
    for face in faces:
        left_eye =[]
        right_eye=[]
        landmarks = predictor(frame,face)
        for n in range(37,43):
            x,y = landmarks.part(n)
            left_eye.append([x,y])
            if n==42:
                next_pt = 37
            else:
                next_pt = n+1
            nx,ny = landmarks.part(next_pt)
            cv2.line(frame,(x,y),(nx,ny),(0,255,0),2)
        for n in range(43,49):
            x,y = landmarks.part(n)
            right_eye.append([x,y])
            if n==48:
                next_pt = 43
            else:
                next_pt = n+1
            nx,ny = landmarks.part(next_pt)
            cv2.line(frame,(x,y),(nx,ny),(0,255,0),2)
    
    left_eye_ratio = eye_aspect_ratio(left_eye)
    right_eye_ratio = eye_aspect_ratio(right_eye)
    avg = left_eye_ratio+right_eye_ratio/2
    avg = round(avg,2)
    if avg<0.26:
        cv2.putText(frame,'Are you Sleepy',(20,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(255,45,215),3)
    else:
        cv2.putText(frame,'NOT SLEEPING BOY',(20,800),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(255,45,215),3)
    
    cv2.imshow('Drowiness System',frame)
    if cv2.waitKey(10)&0xFF==27:
        break
        
capture.release()
cv2.destroyAllWindows
    