import cv2 as cv
import numpy as np
import time

videoCapture = cv.VideoCapture(0)
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1*x2)**2*(y1*y2)**2

while True:

    ret, frame = videoCapture.read()
    if not ret: break

#####

    #height, width, _ = frame.shapeq
    #cx = int(width /2)
    #cy = int(height /2)

    #pixel_center2 = frame[cx, cy]
    #print("BGR = ",  pixel_center2) #BGR color from center circle

    #cv.circle(frame, (cx, cy), 5, (255,0,0),3) #Circle in the center of the video

#####


    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17,17), 0)
    #bgrFrame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 1,
                              param1=60,
                              param2=32,
                              minRadius=30,
                              maxRadius=45)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    chosen = i


        cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)


        pixel_center = frame[chosen[0], chosen[1]]
        cords1 = str(chosen[0])+"-"+str(chosen[1])
        cv.putText(frame, str(cords1), (10, 50), 0, 1, (255,0,0),2)
        print("Pix1 = ", pixel_center)


        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0, 100, 100), 3)
        prevCircle = chosen

    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'): break

    #time.sleep(.5)

videoCapture.release()
cv.destroyAllWindows()
