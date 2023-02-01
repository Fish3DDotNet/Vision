import cv2 as cv
import numpy as np


def main():
    while True:
        videoCapture = cv.VideoCapture(1)
        #videoCapture.set(3, 1280)
        #videoCapture.set(4, 720)
        ret, frame = videoCapture.read()
        if not ret: break

        #frame = cv.resize(frame, (320, 240))
        cv.imshow("frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()