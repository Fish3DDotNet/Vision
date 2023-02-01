import cv2 as cv
import numpy as np
import time

img = cv.VideoCapture(0)
#img = cv.imread("Pics//red.jpg", 1)

# height, width, _ = img.shape
# cx = int(width / 2)
# cy = int(height / 2)
#
# pixel_center2 = img[cy, cx]
# print("BGR = ", pixel_center2)  # BGR color from center circle
while True:
    ret, frame = img.read()
    if not ret: break

    prevCircle = None
    dist = lambda x1,y1,x2,y2: (x1*x2)**2*(y1*y2)**2



    grayFrame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17,17), 0)
    #bgrFrame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 1,
                              param1=60,
                              param2=32,
                              minRadius=5,
                              maxRadius=45)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    chosen = i



    pixel_center = img[240,320]
    #pixel_center = img[chosen[1], chosen[0]]
    print("Pix1 = ", pixel_center)

    cords1 = str(chosen[0])+"-"+str(chosen[1])
    cv.putText(img, str(cords1), (10, 50), 0, .5, (255,255,255),1)
    cv.putText(img, str(pixel_center), (10, 100), 0, .5, (255, 255, 255), 1)

    cv.circle(img, (chosen[0], chosen[1]), 1, (0, 100, 100), 1)

    cv.circle(img, (chosen[0], chosen[1]), chosen[2], (0, 100, 100), 1)

    prevCircle = chosen

    cv.imshow("frame", img)

cv.waitKey(0)
cv.destroyAllWindows()
