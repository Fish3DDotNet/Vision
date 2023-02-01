import cv2 as cv
import numpy as np

def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        _, frame = cap.read()

        height, width, _ = frame.shape

        cx = int(width /2)
        cy = int(height /2)

        pixel_center2 = frame[cy, cx]
        print(pixel_center2)
        cv.circle(frame, (cx, cy), 5, (255,0,0),3)

        cv.imshow("Frame", frame)
        key = cv.waitKey(1)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv.destroyAllWindows()





if __name__ == '__main__':
    main()
