import numpy as np
import cv2
from collections import deque

def nothing(x):
    pass

# ================= TRACKBAR WINDOW =================
cv2.namedWindow("Color Detectors")

cv2.createTrackbar("Upper Hue","Color Detectors",153,180,nothing)
cv2.createTrackbar("Upper Saturation","Color Detectors",255,255,nothing)
cv2.createTrackbar("Upper Value","Color Detectors",255,255,nothing)

cv2.createTrackbar("Lower Hue","Color Detectors",64,180,nothing)
cv2.createTrackbar("Lower Saturation","Color Detectors",72,255,nothing)
cv2.createTrackbar("Lower Value","Color Detectors",49,255,nothing)

# ================= INITIALIZATION =================
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = green_index = red_index = yellow_index = 0

colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorIndex = 0

kernel = np.ones((5,5),np.uint8)

# ================= PAINT WINDOW =================
paintWindow = np.ones((471,636,3), dtype=np.uint8) * 255

def drawButtons(img):
    cv2.rectangle(img,(40,1),(140,65),(0,0,0),2)
    cv2.rectangle(img,(160,1),(255,65),colors[0],-1)
    cv2.rectangle(img,(275,1),(370,65),colors[1],-1)
    cv2.rectangle(img,(390,1),(485,65),colors[2],-1)
    cv2.rectangle(img,(500,1),(600,65),colors[3],-1)
    cv2.putText(img,"CLEAR",(49,33),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

drawButtons(paintWindow)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3, 636)   # width
cap.set(4, 471)   # height
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    drawButtons(frame)

    # Get HSV values
    u_h = cv2.getTrackbarPos("Upper Hue","Color Detectors")
    u_s = cv2.getTrackbarPos("Upper Saturation","Color Detectors")
    u_v = cv2.getTrackbarPos("Upper Value","Color Detectors")

    l_h = cv2.getTrackbarPos("Lower Hue","Color Detectors")
    l_s = cv2.getTrackbarPos("Lower Saturation","Color Detectors")
    l_v = cv2.getTrackbarPos("Lower Value","Color Detectors")

    Upper = np.array([u_h,u_s,u_v])
    Lower = np.array([l_h,l_s,l_v])

    mask = cv2.inRange(hsv,Lower,Upper)
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.dilate(mask,kernel,iterations=1)

    contours,_ = cv2.findContours(mask,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    center = None

    if len(contours) > 0:

        cnt = max(contours,key=cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)

        if radius > 5:

            cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                center = (int(M["m10"]/M["m00"]),
                          int(M["m01"]/M["m00"]))

            if center:

                # ===== BUTTON AREA =====
                if center[1] <= 65:

                    if 40 <= center[0] <= 140:
                        bpoints = [deque(maxlen=1024)]
                        gpoints = [deque(maxlen=1024)]
                        rpoints = [deque(maxlen=1024)]
                        ypoints = [deque(maxlen=1024)]
                        blue_index = green_index = red_index = yellow_index = 0
                        paintWindow[67:, :, :] = 255

                    elif 160 <= center[0] <= 255:
                        colorIndex = 0
                    elif 275 <= center[0] <= 370:
                        colorIndex = 1
                    elif 390 <= center[0] <= 485:
                        colorIndex = 2
                    elif 500 <= center[0] <= 600:
                        colorIndex = 3

                # ===== DRAWING AREA =====
                else:

                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)

    else:
        bpoints.append(deque(maxlen=1024))
        gpoints.append(deque(maxlen=1024))
        rpoints.append(deque(maxlen=1024))
        ypoints.append(deque(maxlen=1024))
        blue_index += 1
        green_index += 1
        red_index += 1
        yellow_index += 1

    # ===== DRAW LINES =====
    points = [bpoints,gpoints,rpoints,ypoints]

    for i in range(4):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame,points[i][j][k-1],points[i][j][k],colors[i],2)
                cv2.line(paintWindow,points[i][j][k-1],points[i][j][k],colors[i],2)

    cv2.imshow("Live Drawing",frame)
    cv2.imshow("Paint",paintWindow)
    cv2.imshow("Mask",mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()