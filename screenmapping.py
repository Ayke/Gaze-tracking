import os
import time
import cv2
import dlib
from scipy import ndimage
import numpy as np
from imutils import face_utils

blocks = 2

canvas_size = (720, 960)
canvas = np.zeros(np.append(canvas_size,3), dtype = "uint8")
hori_break = [a*canvas.shape[1]/blocks for a in range(0,4)]
verti_break = [a*canvas.shape[0]/blocks for a in range(0,4)]
thickness_breaking = 3

center = tuple(a / 2 for a in canvas_size)
eye_center = center
scalar = 1

def activate_block(canvas, number):
    print "Activating " + str(number)
    canvas.fill(0)
    # Slice the canvas into blocks*blocks
    # Vertical lines
    for i in hori_break:
        cv2.line(canvas, (i, 0), (i, canvas.shape[0]), (255,255,255), thickness_breaking)
    # Horizontal lines
    for j in verti_break:
        cv2.line(canvas, (0, j), (canvas.shape[1], j), (255,255,255), thickness_breaking)
    # cv2.imshow("Canvas0", canvas)
    # cv2.waitKey(0)
    if number < 0 or number >= blocks*blocks:
        return -1
    i = number % blocks
    j = number / blocks
    cv2.rectangle(canvas, (hori_break[i] + thickness_breaking, verti_break[j] + thickness_breaking), 
        (hori_break[i+1] - thickness_breaking, verti_break[j+1] - thickness_breaking), (255,0,0), 7)
    return number

# Activate certain position
def activate(point):
    global canvas
    activate_block(canvas, blocks*(point[1] / verti_break[1]) + (point[0] / hori_break[1]))

# Activate when pupil is at location point
def activate_pupil(point, eye_center):
    activate(np.add(np.subtract(point, eye_center)*scalar, center))

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        activate_pupil((x,y), eye_center)


# activate_block(canvas, -1)
# cv2.imshow("Canvas", canvas)
# cv2.waitKey(0)
# for i in range(9):
#     activate_block(canvas, i)
#     cv2.imshow("Canvas", canvas)
#     cv2.waitKey(0)


cv2.namedWindow("canvas")
cv2.setMouseCallback("canvas", click)
activate_block(canvas, -1)

while True:
    cv2.imshow("canvas", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break