import cv2
import os

cap = cv2.VideoCapture(0)

directory = 'Data/'

while True:
    _, frame = cap.read()
    
    
    count = {
        'a': len(os.listdir(os.path.join(directory, 'A'))),
        'b': len(os.listdir(os.path.join(directory, 'B'))),
        'c': len(os.listdir(os.path.join(directory, 'C')))  
    }
    
    row = frame.shape[1]
    col = frame.shape[0]
    
    #draw a rectangle on the frame
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    
    #display capture region separately
    cv2.imshow('Data', frame)
    cv2.imshow('ROI', frame[40:400, 0:300])
    
    frame = frame[40:400, 0:300]
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'A/' + str(count['a']) + '.png', frame)
    elif interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'B/' + str(count['b']) + '.png', frame)
    elif interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory + 'C/' + str(count['c']) + '.png', frame)
        
# release the video capture device
cap.release()
cv2.destroyAllWindows()