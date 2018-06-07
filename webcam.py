import os
import time
import cv2
import dlib
from scipy import ndimage
import numpy as np
from imutils import face_utils

filename = 'rightturn'
FACE_LIB = "./library/haarcascade_frontalface_default.xml"
EYE_LIB = "./library/haarcascade_eye.xml"
GLASS_EYE_LIB = "./library/haarcascade_eye_tree_eyeglasses.xml"
DLIB_LIB = "./library/shape_predictor_68_face_landmarks.dat"


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
blink_threshold = 9

blocks = 3

canvas_size = (720, 960)
canvas = np.zeros(np.append(canvas_size,3), dtype = "uint8")
hori_break = [a*canvas.shape[1]/blocks for a in range(0,blocks+1)]
verti_break = [a*canvas.shape[0]/blocks for a in range(0,blocks+1)]
thickness_breaking = 3

center = (canvas_size[1] / 2, canvas_size[0] / 2)
center_point = center
eye_center = center
# scalar = (-40,100) #(leftRight, upDown)
scalar = (-40,70) #(leftRight, upDown)

def activate_block(canvas, number):
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


# Activate certain position regarding to the center of picture
def activate(point):
    global canvas
    if point[0] > canvas_size[1]:
        point[0] = canvas_size[1]
    if point[0] < 0:
        point[0] = 0
    if point[1] > canvas_size[0]:
        point[1] = canvas_size[1]
    if point[1] < 0:
        point[1] = 0
    print np.subtract(point, center)
    activate_block(canvas, blocks*(point[1] / verti_break[1]) + (point[0] / hori_break[1]))

# Activate when pupil is at certain position regarding to the eye position 
# staring at the center of picture
def activate_pupil(point):
    activate(np.add(
        np.multiply(np.subtract(point, eye_center),scalar),
        center))

# def click(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         activate_pupil((x,y), eye_center)

def headpose(shape):
    mid_x = [(shape.part(1).x+shape.part(15).x)/2,
             (shape.part(1).y+shape.part(15).y)/2]
    mid_y = [(shape.part(27).x+shape.part(66).x)/2,
             (shape.part(27).y+shape.part(66).y)/2]
    nose = [shape.part(30).x, shape.part(30).y]
    final_x = 3*nose[0]-2*mid_x[0]
    final_y = 3*nose[1]-2*mid_y[1]

    cv2.circle(frame, (int(final_x), int(final_y)), 2, (0, 0, 255))
    cv2.circle(frame, (int(nose[0]), int(nose[1])), 2, (0, 0, 255))
    cv2.line(frame, (int(nose[0]), int(nose[1])),
             (int(final_x), int(final_y)), (255, 0, 0), 20)
    return [nose, [final_x, final_y]]


def faces(landmark):
    facial = face_utils.shape_to_np(landmark)
    for (x, y) in facial:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), 10)
    return facial


def track(old_gray, gray, irises, blinks, blink_in_previous):
    lost_track = False
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, gray, irises, None, **lk_params)
    if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
        lost_track = True
        blink_in_previous = False
    # high error rate in klt tracking
    elif err[0][0] > blink_threshold or err[1][0] > blink_threshold:
        lost_track = True
        if not blink_in_previous:
            blinks += 1
            blink_in_previous = True
    else:
        blink_in_previous = False
        irises = []
        for w, h in p1:
            irises.append([w, h])
        irises = np.array(irises)
    return irises, blinks, blink_in_previous, lost_track

def get_irises_location(eye):
    pass

def first_regulate(landmark):
    global eye_center

    landmark = face_utils.shape_to_np(landmark)
    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    left_eye = np.float32(landmark[j:k])
    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    right_eye = np.float32(landmark[j:k])

    (lcx, lcy) = (0.0, 0.0)
    (rcx, rcy) = (0.0, 0.0)
    if len(right_eye) > 0:
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
            rcx += x; rcy += y
        rcx /= len(right_eye); rcy /= len(right_eye)
        rcx = int(rcx); rcy = int(rcy)
    if len(left_eye) > 0:
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
            lcx += x; lcy += y
        lcx /= len(left_eye); lcy /= len(left_eye)
        lcx = int(lcx); lcy = int(lcy)

    x, y, w, h = cv2.boundingRect(right_eye)
    right_eye_frame = frame[y:(y+h), x:(x+w)]
    if right_eye_frame.shape[0] > 0 and right_eye_frame.shape[1] > 0:
        gray_right_eye_frame = cv2.cvtColor(
            right_eye_frame, cv2.COLOR_BGR2GRAY)
        (th, thc) = (35, 19)
        binary_right_eye_frame = cv2.adaptiveThreshold(gray_right_eye_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, th, thc)
        _, contours, _ = cv2.findContours(binary_right_eye_frame,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        i = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1e-5:
                del contours[i]
                break
            i += 1

        # Pick the largest blob as the right eye
        bestBlob = -1
        if len(contours) >= 2:
            maxArea = -1
            maxIndex = 0
            i = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > maxArea:
                    maxArea = cv2.contourArea(cnt)
                    maxIndex = i
                i += 1
            bestBlob = maxIndex
        elif len(contours) == 1:
            bestBlob = 0
        else:
            bestBlob = -1

        if bestBlob >= 0:	
            center = cv2.moments(contours[bestBlob])
            if center['m00'] == 0:
                (cx,cy) = (0,0)
                print "Regulate Error!!!!"
            else:
                (cx,cy) = (int(center['m10']/center['m00']), int(center['m01']/center['m00']))
            eye_center = (x+cx-rcx, y+cy-rcy)
            cv2.circle(right_eye_frame,(cx,cy),3,(0,255,0),1)
            cv2.circle(right_eye_frame,(rcx-x,rcy-y),1,(255,255,0),1)
        else:
            print "Regulate Error!!!!"
    cv2.imshow("first reg", cv2.resize(right_eye_frame, (0, 0), fx=10, fy=10))


def second_regulate(landmark):
    global eye_center
    print eye_center
    global scalar

    landmark = face_utils.shape_to_np(landmark)
    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    left_eye = np.float32(landmark[j:k])
    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    right_eye = np.float32(landmark[j:k])

    (lcx, lcy) = (0.0, 0.0)
    (rcx, rcy) = (0.0, 0.0)
    if len(right_eye) > 0:
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
            rcx += x; rcy += y
        rcx /= len(right_eye); rcy /= len(right_eye)
        rcx = int(rcx); rcy = int(rcy)
    if len(left_eye) > 0:
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
            lcx += x; lcy += y
        lcx /= len(left_eye); lcy /= len(left_eye)
        lcx = int(lcx); lcy = int(lcy)

    x, y, w, h = cv2.boundingRect(right_eye)
    right_eye_frame = frame[y:(y+h), x:(x+w)]
    if right_eye_frame.shape[0] > 0 and right_eye_frame.shape[1] > 0:
        gray_right_eye_frame = cv2.cvtColor(
            right_eye_frame, cv2.COLOR_BGR2GRAY)
        (th, thc) = (35, 19)
        binary_right_eye_frame = cv2.adaptiveThreshold(gray_right_eye_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, th, thc)
        _, contours, _ = cv2.findContours(binary_right_eye_frame,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        i = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1e-5:
                del contours[i]
                break
            i += 1

        # Pick the largest blob as the right eye
        bestBlob = -1
        if len(contours) >= 2:
            maxArea = -1
            maxIndex = 0
            i = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > maxArea:
                    maxArea = cv2.contourArea(cnt)
                    maxIndex = i
                i += 1
            bestBlob = maxIndex
        elif len(contours) == 1:
            bestBlob = 0
        else:
            bestBlob = -1

        if bestBlob >= 0:	
            center = cv2.moments(contours[bestBlob])
            if center['m00'] == 0:
                (cx,cy) = (0,0)
                print "Regulate Error!!!!"
            else:
                (cx,cy) = (int(center['m10']/center['m00']), int(center['m01']/center['m00']))
            print (x+cx-rcx, y+cy-rcy)
            # tmp = np.true_divide(
            #     center_point,
            #     np.subtract((x+cx-rcx, y+cy-rcy), eye_center)
            # )
            # print tmp
            # scalar = tmp
            cv2.circle(right_eye_frame,(cx,cy),3,(0,255,0),1)
            cv2.circle(right_eye_frame,(rcx-x,rcy-y),1,(255,255,0),1)
            
        else:
            print "Regulate Error!!!!"
    cv2.imshow("second reg", cv2.resize(right_eye_frame, (0, 0), fx=10, fy=10))



def eyes(gray, old_gray, irises, blinks, blink_in_previous, landmark):
    landmark = face_utils.shape_to_np(landmark)
    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    left_eye = np.float32(landmark[j:k])
    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    right_eye = np.float32(landmark[j:k])
    

    # left_pupil_x -> lcx
    # right_pupil_x -> rcx
    (lpx, lpy) = (0.0, 0.0)
    (rpx, rpy) = (0.0, 0.0)

    # left_center_x -> lcx
    # right_center_x -> rcx
    (lcx, lcy) = (0.0, 0.0)
    (rcx, rcy) = (0.0, 0.0)
    if len(right_eye) > 0:
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
            rcx += x; rcy += y
        rcx /= len(right_eye); rcy /= len(right_eye)
        rcx = int(rcx); rcy = int(rcy)
    if len(left_eye) > 0:
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
            lcx += x; lcy += y
        lcx /= len(left_eye); lcy /= len(left_eye)
        lcx = int(lcx); lcy = int(lcy)
    # cv2.circle(frame, (int(lcx), int(lcy)), 1, (255, 255, 255))
    # cv2.circle(frame, (int(rcx), int(rcy)), 1, (255, 255, 255))

    x, y, w, h = cv2.boundingRect(right_eye)
    right_eye_frame = frame[y:(y+h), x:(x+w)]
    if right_eye_frame.shape[0] > 0 and right_eye_frame.shape[1] > 0:
        gray_right_eye_frame = cv2.cvtColor(
            right_eye_frame, cv2.COLOR_BGR2GRAY)
        
        # (th, thc) = (15, 7)
        # (th, thc) = (25, 14)
        (th, thc) = (25, 19)
        (th, thc) = (35, 19) #Very nice for now, Tuesday
        # (th, thc) = (45, 42)
        # (th, thc) = (55, 34)
        # (th, thc) = (65, 34)
        # (th, thc) = (75, 50)


        ### Saved for testing options in adaptiveThreshold ########
        # cv2.imwrite("right_mean_sample.jpg", gray_right_eye_frame)
        # for th in [25,35,45,55,65,75,85,95]:
        #     for thc in range(1, 50, 3):
        #         binary_gray_right_eye_frame = cv2.adaptiveThreshold(gray_right_eye_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, th, thc)
        #         # binary_gray_right_eye_frame = cv2.adaptiveThreshold(gray_right_eye_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, th, thc)
        #         # cv2.imshow("right eye " + str(th) + " " +str(thc), cv2.resize(binary_gray_right_eye_frame, (0, 0), fx=10, fy=10))
        #         cv2.imwrite("right_" + "mean_" + str(th) + "_" + str(thc) + ".jpg", cv2.resize(binary_gray_right_eye_frame, (0, 0), fx=10, fy=10))
        # print "OK byebye!!!!"
        # while True:
        #     if th == -1:
        #         th = 0
        ############################################################
        

        # Use Adaptive Mean Method to binarize image, eyes will be white (255)
        binary_right_eye_frame = cv2.adaptiveThreshold(gray_right_eye_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, th, thc)
        _, contours, _ = cv2.findContours(binary_right_eye_frame,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        i = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1e-5:
                del contours[i]
                break
            i += 1

        # Pick the largest blob as the right eye
        bestBlob = -1
        if len(contours) >= 2:
            maxArea = -1
            maxIndex = 0
            i = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > maxArea:
                    maxArea = cv2.contourArea(cnt)
                    maxIndex = i
                i += 1
            bestBlob = maxIndex
        elif len(contours) == 1:
            bestBlob = 0
        else:
            bestBlob = -1

        if bestBlob >= 0:	
            center = cv2.moments(contours[bestBlob])
            if center['m00'] == 0:
                (cx,cy) = (0,0)
            else:
                (cx,cy) = (int(center['m10']/center['m00']), int(center['m01']/center['m00']))
            cv2.circle(right_eye_frame,(cx,cy),3,(0,255,0),1)
            cv2.circle(right_eye_frame,(rcx-x,rcy-y),1,(255,255,0),1)
            # cv2.circle(frame,(x+cx,y+cy),3,(0,255,0),1)
            activate_pupil((x+cx-rcx, y+cy-rcy))

        cv2.imshow("frame", frame)
        # cv2.imshow("right eye binary", cv2.resize(binary_right_eye_frame, (0, 0), fx=10, fy=10))
        # for i in range(len(contours)):
        #     cv2.drawContours(right_eye_frame, contours, i, ((i % 3 ==0)*255,(i % 3 == 1)*255, (i % 3 == 2)*255))
        cv2.imshow("right eye", cv2.resize(right_eye_frame, (0, 0), fx=10, fy=10))




    x, y, w, h = cv2.boundingRect(left_eye)
    left_eye_frame = frame[y:(y+h), x:(x+w)]
    if left_eye_frame.shape[0] > 0 and left_eye_frame.shape[1] > 0:
        left_eye_frame = cv2.resize(left_eye_frame, (0, 0), fx=10, fy=10)

    # if len(irises) >= 2:  # irises detected, track eyes
    #     track_result = track(old_gray, gray, irises, blinks, blink_in_previous)
    #     irises, blinks, blink_in_previous, lost_track = track_result
    #     if lost_track:
    #         irises = get_irises_location(gray)
    # else:
    #     irises = get_irises_location(gray)

    for (x, y) in irises:
        cv2.circle(frame, (x, y), 1, (0, 255, 0))
    return irises, blinks, blink_in_previous, left_eye, right_eye


def save_to_file(facial, pose, irises, files):
    if(len(irises) == 2):
        for f in facial:
            for x in f:
                files.write(str(x)+' ')
        files.write('\n')
        for p in pose:
            for x in p:
                files.write(str(x)+' ')
        files.write('\n')
        for i in irises:
            for x in i:
                files.write(str(x)+' ')
        files.write('\n')
        files.write('\n')


if __name__ == "__main__":
    print("[INFO] loading predictors...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_LIB)
    face_cascade = cv2.CascadeClassifier(FACE_LIB)
    eye_cascade = cv2.CascadeClassifier(EYE_LIB)
    print("[INFO] camera sensor warming up...")
    webcam = cv2.VideoCapture(0)
    time.sleep(2.0)

    # vairable
    irises = []
    old_gray = []
    blinks = 0
    blink_in_previous = False
    # files = open(filename + '.txt','w')

    #photos = os.listdir('./photos/')
    # print(photos)
    # for photo in photos:




    print("[INFO] regulate eye contact")
    ##First regulation
    activate_block(canvas, -1)
    cv2.circle(canvas,center, 3, (0,0,255), 7)
    cv2.imshow("canvas", canvas)
    cv2.waitKey(0)
    
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 1:
        print "Error while first regulating!!"
    for rect in rects:
        landmark = predictor(gray, rect)
        first_regulate(landmark)

    ##Second regulation
    activate_block(canvas, -1)
    cv2.circle(canvas, (0,0), 3, (0,0,255), 7)
    cv2.imshow("canvas", canvas)
    cv2.waitKey(0)

    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 1:
        print "Error while second regulating!!"
    for rect in rects:
        landmark = predictor(gray, rect)
        second_regulate(landmark)



    while True:
        ret, frame = webcam.read()
        #frame = cv2.imread('./photos/'+photo)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            landmark = predictor(gray, rect)
            # facial = faces(landmark)
            # pose = headpose(landmark)
            irises, blinks, blink_in_previous, left_eye, right_eye = eyes(
                gray, old_gray, irises, blinks, blink_in_previous, landmark)
            # save_to_file(facial, pose, irises, files)

        frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
        resized_image = cv2.flip(frame, 1)
        # cv2.imshow('Frame', resized_image)
        old_gray = gray.copy()
        # cv2.resizeWindow('Frame', 960,720)


        cv2.imshow("canvas", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
