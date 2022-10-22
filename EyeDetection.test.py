import cv2
from EyeDetection import detectEyes

cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()
    eyes = detectEyes(img,True)
    print(eyes)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()