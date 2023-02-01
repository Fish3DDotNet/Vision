#https://pysource.com/2021/10/19/simple-color-recognition-with-opencv-and-python/
import cv2
import numpy as np


def nothing(x):
    pass

# Trackbar
cv2.namedWindow("ui")
cv2.createTrackbar("H", "ui", 0, 179, nothing)
cv2.createTrackbar("S", "ui", 255, 255, nothing)
cv2.createTrackbar("V", "ui", 255, 255, nothing)

img_hsv = np.zeros((250, 500, 3), np.uint8)

while True:
    h = cv2.getTrackbarPos("H", "ui")
    s = cv2.getTrackbarPos("S", "ui")
    v = cv2.getTrackbarPos("V", "ui")

    img_hsv[:] = (h, s, v)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("ui", img_bgr)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

