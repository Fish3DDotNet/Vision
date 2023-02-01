import cv2 as cv
import numpy as np
import time


def nothing(x):
    # pixel_center = frame[y, x]
    # location = str(x)+'-'+str(y)
    # b1, g1, r1 = int(pixel_center[0][0]), int(pixel_center[0][1]), int(pixel_center[0][2])
    # b2, g2, r2 = int(pixel_center2[0]), int(pixel_center2[1]), int(pixel_center2[2])
    # cords1 = str(chosen[0])+"-"+str(chosen[1])
    # cv.putText(frame, str(pixel_center[0]), (10, 50), 0, 1, (b1, g1, r1),2)
    # cv.putText(frame, str(location), (10, 80), 0, 1, (b2, g2, r2), 2)
    # cv.putText(frame, str(pixel_center2), (10, 110), 0, 1, (b2, g2, r2), 2)
    # cv.putText(frame, str(rad), (x+30,y-30), 0, 1, (b1, g1, r1), 2)

    ################################################################################################################

    # height, width, _ = frame.shape
    # cx = int(width /2)
    # cy = int(height /2)
    #
    # pc2 = frame[cy, cx]
    # print("BGR = ",  pc2) #BGR color from center circle
    #
    # cv.circle(frame, (cx, cy), 5, (255,0,0),1) #Circle in the center of the video

    ################################################################################################################
    pass

def zoomVideo(image, Iscale=1):
    try:
        scale=Iscale

        #get the webcam size
        height, width, channels = image.shape

        #prepare the crop
        centerX,centerY=int(height/2),int(width/2)
        radiusX,radiusY= int(scale*centerX),int(scale*centerY)

        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY

        cropped = image[minX:maxX, minY:maxY]
        resized_cropped = cv.resize(cropped, (width, height))

        return resized_cropped

    except Exception as e:
        x = str(e)

        return image

def drwtxt():#draw text on screen
    pass

def locations(percent, radius):
    rp = radius * percent
    rc = rp * 0.707
    return rp, rc

def getpxls(frame, y, x, rad):
    pixelArray = []
    rad = 30

    pixelArray.append(frame[y, x]) #center

    rp, rc = locations(.9, rad)

    pixelArray.append(frame[int(y-rp), x])             # 0
    pixelArray.append(frame[int(y-rc), int(x+rc)])    # 45
    pixelArray.append(frame[y, int(x+rp)])             # 90
    pixelArray.append(frame[int(y+rc), int(x+rc)])    # 135
    pixelArray.append(frame[int(y+rp), x])             # 180
    pixelArray.append(frame[int(y+rc), int(x-rc)])    # 225
    pixelArray.append(frame[y, int(x-rp)])             # 270
    pixelArray.append(frame[int(y-rc), int(x-rc)])    # 315

    rp, rc = locations(.7, rad)

    pixelArray.append(frame[int(y-rp), x])             # 0
    pixelArray.append(frame[int(y-rc), int(x+rc)])    # 45
    pixelArray.append(frame[y, int(x+rp)])             # 90
    pixelArray.append(frame[int(y+rc), int(x+rc)])    # 135
    pixelArray.append(frame[int(y+rp), x])             # 180
    pixelArray.append(frame[int(y+rc), int(x-rc)])    # 225
    pixelArray.append(frame[y, int(x-rp)])             # 270
    pixelArray.append(frame[int(y-rc), int(x-rc)])    # 315

    rp, rc = locations(.5, rad)

    pixelArray.append(frame[int(y-rp), x])             # 0
    pixelArray.append(frame[int(y-rc), int(x+rc)])    # 45
    pixelArray.append(frame[y, int(x+rp)])             # 90
    pixelArray.append(frame[int(y+rc), int(x+rc)])    # 135
    pixelArray.append(frame[int(y+rp), x])             # 180
    pixelArray.append(frame[int(y+rc), int(x-rc)])    # 225
    pixelArray.append(frame[y, int(x-rp)])             # 270
    pixelArray.append(frame[int(y-rc), int(x-rc)])    # 315

    rp, rc = locations(.3, rad)

    pixelArray.append(frame[int(y-rp), x])             # 0
    pixelArray.append(frame[int(y-rc), int(x+rc)])    # 45
    pixelArray.append(frame[y, int(x+rp)])             # 90
    pixelArray.append(frame[int(y+rc), int(x+rc)])    # 135
    pixelArray.append(frame[int(y+rp), x])             # 180
    pixelArray.append(frame[int(y+rc), int(x-rc)])    # 225
    pixelArray.append(frame[y, int(x-rp)])             # 270
    pixelArray.append(frame[int(y-rc), int(x-rc)])    # 315

    return pixelArray

def getavg(pc):
    av=[0,0,0]
    for i in pc:
        av[0] = av[0] + i[0]
        av[1] = av[1] + i[1]
        av[2] = av[2] + i[2]
    av[0] = int(av[0] / len(pc))
    av[1] = int(av[1] / len(pc))
    av[2] = int(av[2] / len(pc))

    return av

def getminmax(pcrange, pcavg):

    if pcavg[0]<pcrange[0][0]:
        pcrange[0][0]=pcavg[0]
    if pcavg[0]>pcrange[0][1]:
        pcrange[0][1]=pcavg[0]

    if pcavg[1]<pcrange[1][0]:
        pcrange[1][0]=pcavg[1]
    if pcavg[1]>pcrange[1][1]:
        pcrange[1][1]=pcavg[1]

    if pcavg[2]<pcrange[2][0]:
        pcrange[2][0]=pcavg[2]
    if pcavg[2]>pcrange[2][1]:
        pcrange[2][1]=pcavg[2]

    return pcrange

def drawrec(frame, pc):

    i=0
    cv.rectangle(frame, (20, 20), (40, 40), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1);i+=1  # 0
    cv.rectangle(frame, (20, 0), (40, 20), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1);i+=1  # 1
    cv.rectangle(frame, (40, 0), (60, 20), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1);i+=1  # 2
    cv.rectangle(frame, (40, 20), (60, 40), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1);i+=1  # 3   8 1 2
    cv.rectangle(frame, (40, 40), (60, 60), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1);i+=1  # 4   7 0 3
    cv.rectangle(frame, (20, 40), (40, 60), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1);i+=1  # 5   6 5 4
    cv.rectangle(frame, (0, 40), (20, 60), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1);i+=1  # 6
    cv.rectangle(frame, (0, 20), (20, 40), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1);i+=1 # 7
    cv.rectangle(frame, (0, 0), (20, 20), (int(pc[i][0]), int(pc[i][1]), int(pc[i][2])), -1)  # 8

def draw_gradient_alpha_rectangle(frame, BGR_Channel, rectangle_position, rotate):
    (xMin, yMin), (xMax, yMax) = rectangle_position
    color = np.array(BGR_Channel, np.uint8)[np.newaxis, :]
    mask1 = np.rot90(np.repeat(np.tile(np.linspace(1, 0, (rectangle_position[1][1]-rectangle_position[0][1])), ((rectangle_position[1][0]-rectangle_position[0][0]), 1))[:, :, np.newaxis], 3, axis=2), rotate)
    frame[yMin:yMax, xMin:xMax, :] = mask1 * frame[yMin:yMax, xMin:xMax, :] + (1-mask1) * color

    return frame

def gradient(pcrange):
    gradientFrame = np.zeros((400, 400, 3), np.uint8)
    gradientFrame[:,:,:] = int(pcrange[0][0]),int(pcrange[1][0]),int(pcrange[2][0])

    gradientFrame = draw_gradient_alpha_rectangle(gradientFrame, (int(pcrange[0][1]),int(pcrange[1][1]),int(pcrange[2][1])), ((0, 0), (400, 400)), 2)

    cv.imshow('gradientFrame', gradientFrame)

def main():

    videoCapture = cv.VideoCapture(1)
    prevCircle = None
    dist = lambda x1, y1, x2, y2: (x1 * x2) ** 2 * (y1 * y2) ** 2
    pc = []
    green = np.uint8([[[0, 100, 0]]])
    pcrange = [255, 0], [255, 0], [255, 0]

    # Trackbar
    cv.namedWindow("ui")
    cv.createTrackbar("param1", "ui", 100, 300, nothing)
    cv.createTrackbar("param2", "ui", 70, 300, nothing)
    cv.createTrackbar("minRadius", "ui", 30, 100, nothing)
    cv.createTrackbar("maxRadius", "ui", 50, 100, nothing)
    #cv.createTrackbar("zoom", "ui", 1.0, 1.0, nothing)

    img_hsv = np.zeros((250, 500, 3), np.uint8)

    while True:

        ret, frame = videoCapture.read()
        if not ret: break

        param1 = cv.getTrackbarPos("param1", "ui")
        param2 = cv.getTrackbarPos("param2", "ui")
        minRadius = cv.getTrackbarPos("minRadius", "ui")
        maxRadius = cv.getTrackbarPos("maxRadius", "ui")
        #zmValue = cv.getTrackbarPos("zoom", "ui")

        #### Setup circle detection
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur=13
        blurFrame = cv.GaussianBlur(grayFrame, (int(blur),int(blur)), 1)
        circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 10,
                                  param1=param1,
                                  param2=param2,
                                  minRadius=minRadius,
                                  maxRadius=maxRadius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None
            for i in circles[0, :]:
                if chosen is None: chosen = i
                if prevCircle is not None:
                    if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                        chosen = i

            #cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)

            x,y,rad = chosen[0], chosen[1], chosen[2] #(x,y,radius)

            try:
                # Get pixel colors
                pc = getpxls(frame, y, x, rad)
                pcavg = getavg(pc)
                getminmax(pcrange, pcavg)
                drawrec(frame, pc)

                #print("Pix1 = ", pc[0])
            except:
                pass
            green = np.uint8([[[int(pcavg[0]), int(pcavg[1]), int(pcavg[2])]]])

            cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0, 100, 100), 3)
            prevCircle = chosen
        else:
            pcrange = [255, 0], [255, 0], [255, 0]

        # convert the color to HSV
        hsvGreen = cv.cvtColor(green, cv.COLOR_BGR2HSV)
        hsv = hsvGreen[0][0][0],hsvGreen[0][0][1],hsvGreen[0][0][2]
        cv.putText(frame, str(hsv), (10, 90), 0, 1, (0,0,0), 2)

        img_hsv[:] = hsvGreen[0][0]
        img_bgr = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        cv.imshow("ui", img_bgr)


        # lowerLimit = (83,100,100)#hsvGreen[0][0][0] - 10, 100, 100
        # upperLimit = (133,255,255)#hsvGreen[0][0][0] + 10, 255, 255
        # if (np.greater(hsvGreen[0][0][0],lowerLimit[0]) and np.less(hsvGreen[0][0][0],upperLimit[0])):
        #     print("")
        #     print(lowerLimit)
        #     print(upperLimit)
        #     print(hsvGreen[0][0][0],hsvGreen[0][0][1],hsvGreen[0][0][2])
        #     print(pcavg)

        gradient(pcrange)

        #cropImg = zoomVideo(frame, .5)
        #cv.imshow("cropimg", cropImg)

        cv.imshow("frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        #time.sleep(3)
    videoCapture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()