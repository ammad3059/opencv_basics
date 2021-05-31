import cv2
face_classifier = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
eyes_classifier = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_eye.xml')
def detect_face_eyes(image):
    image = image.copy()
    params = face_classifier.detectMultiScale(image,1.2,1)
    for x,y,w,h in params:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),8)
    
    params2 = eyes_classifier.detectMultiScale(image,1.3,2)
    for x,y,w,h in params2:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),4)
    return image

# video detecting
cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    frame = detect_face_eyes(frame)
    cv2.imshow('Face and Eyes Detection',frame)
    if cv2.waitKey(2) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()