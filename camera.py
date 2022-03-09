import cv2

def take_pic(path,cam_v):
    cap = cv2.VideoCapture(cam_v)
    ret, frame = cap.read()
    cv2.imwrite(path, frame)
