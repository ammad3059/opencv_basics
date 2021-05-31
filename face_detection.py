import cv2
face_classifier = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
def detect_face(image):
    img = image.copy()
    params = face_classifier.detectMultiScale(img)
    for x,y,w,h in params:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
    return img

## video capturing
cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    frame = detect_face(frame)
    cv2.imshow('detecting Face',frame)
    if cv2.waitKey(3) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
