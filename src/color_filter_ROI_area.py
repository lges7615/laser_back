import cv2
import numpy as np
import math

# red_laser = (130,50,150,255,255,255)

cap = cv2.VideoCapture(0)
def nothing(*arg):
        pass

icol = (142, 136, 172, 255, 255, 255)   # Red
icol2 = (1, 1, 1, 2, 3, 90) #erosion & delition
icol3 = (76, 193, 423, 35, 111, 234, 400, 32) # ROI
cv2.namedWindow('colorTest')
cv2.namedWindow('ROI')
cv2.namedWindow('erosion, dilation & canny')
#cv2.namedWindow('line detection')

#icol = (150, 30, 80, 255, 255, 255)    # Green
# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

# ROI
cv2.createTrackbar('x1', 'ROI', icol3[0], 637, nothing)
cv2.createTrackbar('y1', 'ROI', icol3[1], 748, nothing)
cv2.createTrackbar('w1', 'ROI', icol3[2], 637, nothing)
cv2.createTrackbar('h1', 'ROI', icol3[3], 748, nothing)
cv2.createTrackbar('x2', 'ROI', icol3[4], 637, nothing)
cv2.createTrackbar('y2', 'ROI', icol3[5], 748, nothing)
cv2.createTrackbar('w2', 'ROI', icol3[6], 637, nothing)
cv2.createTrackbar('h2', 'ROI', icol3[7], 748, nothing)

# median blur
cv2.createTrackbar('blur weight', 'erosion, dilation & canny', icol2[0], 10, nothing)

# erosion
cv2.createTrackbar('erosion time', 'erosion, dilation & canny', icol2[1], 6, nothing)
cv2.createTrackbar('erosion size', 'erosion, dilation & canny', icol2[2], 4, nothing)
cv2.createTrackbar('dilation time', 'erosion, dilation & canny', icol2[3], 6, nothing)
cv2.createTrackbar('dilation size', 'erosion, dilation & canny', icol2[4], 4, nothing)
cv2.createTrackbar('threshold', 'erosion, dilation & canny', icol2[5], 255, nothing)
'''
# canny
cv2.createTrackbar('low threshold', 'erosion, dilation & canny', icol2[5], 100, nothing)
cv2.createTrackbar('high threshold', 'erosion, dilation & canny', icol2[6], 100, nothing)
'''
# obstruction
front_init = 3260.0
front_thre = 150.0
back_init = 3550.0
back_thre = 150.0

p_f = []
p_b = []

while(1):
    _, frame = cap.read()

    # Get HSV values from the GUI sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
    lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
    lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
    highHue = cv2.getTrackbarPos('highHue', 'colorTest')
    highSat = cv2.getTrackbarPos('highSat', 'colorTest')
    highVal = cv2.getTrackbarPos('highVal', 'colorTest')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([lowHue,lowSat,lowVal])
    upper_red = np.array([highHue,highSat,highVal])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    res = cv2.bitwise_and(frame,frame, mask=mask)

    # median blur
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur_weight = cv2.getTrackbarPos('blur weight', 'erosion, dilation & canny')
    med_blur = cv2.medianBlur(gray,blur_weight)

    # get ersion from GUI sliders.
    erosion_size = cv2.getTrackbarPos('erosion size', 'erosion, dilation & canny')
    e_kernel = np.ones((erosion_size,erosion_size),np.uint8) 
 
    erosion_time = cv2.getTrackbarPos('erosion time', 'erosion, dilation & canny')
    erosion = cv2.erode(med_blur,e_kernel,iterations = erosion_time)


    # get dilation from GUI sliders.

    dilation_size = cv2.getTrackbarPos('dilation size', 'erosion, dilation & canny')
    d_kernel = np.ones((dilation_size,dilation_size),np.uint8) 
 
    dilation_time = cv2.getTrackbarPos('dilation time', 'erosion, dilation & canny')
    dilation = cv2.dilate(erosion,d_kernel,iterations = dilation_time)
    threshold = cv2.getTrackbarPos('threshold', 'erosion, dilation & canny')
    (_, binary) = cv2.threshold(dilation,threshold,255,cv2.THRESH_BINARY)
    '''
    # canny
    low_threshold = cv2.getTrackbarPos('low threshold', 'erosion, dilation & canny')
    high_threshold = cv2.getTrackbarPos('high threshold', 'erosion, dilation & canny')
    edges = cv2.Canny(binary, low_threshold, high_threshold)
    '''
    # ROI
    x1 = cv2.getTrackbarPos('x1', 'ROI')
    y1 = cv2.getTrackbarPos('y1', 'ROI')
    w1 = cv2.getTrackbarPos('w1', 'ROI')
    h1 = cv2.getTrackbarPos('h1', 'ROI')
    x2 = cv2.getTrackbarPos('x2', 'ROI')
    y2 = cv2.getTrackbarPos('y2', 'ROI')
    w2 = cv2.getTrackbarPos('w2', 'ROI')
    h2 = cv2.getTrackbarPos('h2', 'ROI')
    #x1=200
    #y1=120
    #w1=335
    #h1=10

    #x2=245
    #y2=180
    #w2=300
    #h2=10

    #ROI = np.concatenate((ROIfront, ROIback), axis=0)
    ROI = res.copy()
    ROI = cv2.rectangle(ROI, (x1,y1), (x1+w1,y1+h1),[0,255,0], 2)
    ROI = cv2.rectangle(ROI, (x2,y2), (x2+w2,y2+h2),[0,0,255], 2)
    ROI = cv2.circle(ROI, (x1,y1), 6, [0,255,0], -1)
    ROI = cv2.circle(ROI, (x2,y2), 6, [0,0,255], -1)

    ROIfront = binary[y1:y1+h1, x1:x1+w1]  # [y,x]
    ROIback = binary[y2:y2+h2, x2:x2+w2]
    areaf = np.sum(ROIfront)/255
    areab = np.sum(ROIback)/255
    
    p_f = np.append(p_f,areaf)
    p_b = np.append(p_b,areab)
    
    if len(p_f)==20:
        areaf = sum(p_f)//len(p_f)
        areab = sum(p_b)//len(p_b)
        print('areaf',areaf,'areab',areab)
        p_f = []
        p_b = []
    '''
    # obstruction
    if len(p_f)==10:
        areaf = sum(p_f)//len(p_f)
        areab = sum(p_b)//len(p_b)
        if areaf > front_init+front_thre or areaf < front_init-front_thre:
            f = 1
        else:
            f = 0

        if areab > back_init+back_thre or areab < back_init-back_thre:
            b = 1        
        else:
            b = 0
        p_f = []
        p_b = []

        if f==1 and b==1:
            print('STOP!')
        elif f==1 and b==0:
            print('watch out')
        elif f==0 and b==1:
            print('crossing gap')
        else: 
            print('GO!')
    '''
    cv2.imshow('colorTest',res)
    #cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)
    cv2.imshow('erosion, dilation & canny',binary)
    #cv2.imshow('binary',binary)
    cv2.imshow('ROI',ROI)
    cv2.imshow('ROI front', ROIfront)
    cv2.imshow('ROI back', ROIback)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
