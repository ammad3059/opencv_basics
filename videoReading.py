import cv2
import time
path = 'hand_move.mp4'
cap = cv2.VideoCapture(path)
if cap.isOpened() == False:
    print('Error while reading the video file')
else:
    while(cap.isOpened()):    
        ret,frame = cap.read()
        if ret:    
            #frame_heigth = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            #frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            #frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            time.sleep(1//fps)
            cv2.imshow('Video Reading',frame)
            if cv2.waitKey(5)&0xFF==27:
                break
        else:
            print('Error')
            break
    cap.release()
    cv2.destroyAllWindows()