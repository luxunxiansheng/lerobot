import cv2
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)  # 0 and 2 are the camera indexes
ret, frame = cap.read()
if ret:
    print("Camera access successful!")
    print(frame.shape)
else:
    print("Failed to access camera")
cap.release()

