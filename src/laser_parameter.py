#!/usr/bin/env python
# 20191113 WT
# v2.2(to v1.0 laser_subscriber.py)
# v1.0 is named camera_test.py
# use keyboard to publish msg once
# HSV filter, erison and dilation
# press 1, 2 on keyboard can save dual camera parameter
# realize publish parameter

#############   comment   #############
# 1. need to use usb_cam node to subscribe image for dual camera

#######################################

import rospy
import cv2
import numpy as np
from enum import Enum
from std_msgs.msg import String
from laser_back.msg import FilterParameter
        
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap1.set(cv2.CAP_PROP_EXPOSURE,0.1)

cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap2.set(cv2.CAP_PROP_EXPOSURE,0.05)

cv2.namedWindow('color1')
cv2.namedWindow('color2')
#cv2.namedWindow('clearer1')
#cv2.namedWindow('clearer2')

cv2.namedWindow('ROI1')
cv2.namedWindow('ROI2')

def nothing(*arg):
    pass

#################################   HSV   ###############################

# create HSV bar
def HSVbar(FrameName):
    # Lower range colour sliders.
    cv2.createTrackbar('lowHue', FrameName, 1, 255, nothing)
    cv2.createTrackbar('lowSat', FrameName, 1, 255, nothing)
    cv2.createTrackbar('lowVal', FrameName, 1, 255, nothing)
    # Higher range colour sliders.
    cv2.createTrackbar('highHue', FrameName, 255, 255, nothing)
    cv2.createTrackbar('highSat', FrameName, 255, 255, nothing)
    cv2.createTrackbar('highVal', FrameName, 255, 255, nothing)

# Get HSV values from the GUI sliders.
def HSV(FrameName, inputimg):
    global lowHue, lowSat, lowVal, highHue, highSat, highVal
 
    lowHue = cv2.getTrackbarPos('lowHue', FrameName)
    lowSat = cv2.getTrackbarPos('lowSat', FrameName)
    lowVal = cv2.getTrackbarPos('lowVal', FrameName)
    highHue = cv2.getTrackbarPos('highHue', FrameName)
    highSat = cv2.getTrackbarPos('highSat', FrameName)
    highVal = cv2.getTrackbarPos('highVal', FrameName)
    hsv = cv2.cvtColor(inputimg, cv2.COLOR_BGR2HSV)

    lower_red = np.array([lowHue,lowSat,lowVal])
    upper_red = np.array([highHue,highSat,highVal])
	    
    mask = cv2.inRange(hsv, lower_red, upper_red)
		
    res = cv2.bitwise_and(inputimg, inputimg, mask=mask)

    return res
###################################  clear  ##############################

def clearerbar(FrameName):
    cv2.createTrackbar('erosion time', FrameName, 1, 6, nothing)
    cv2.createTrackbar('erosion size', FrameName, 1, 4, nothing)
    cv2.createTrackbar('dilation time', FrameName, 1, 6, nothing)
    cv2.createTrackbar('dilation size', FrameName, 1, 4, nothing)
    cv2.createTrackbar('threshold', FrameName, 1, 255, nothing)

def clearer(FrameName, inputimg):
    global erosion_size, erosion_time, dilation_size, dilation_time, threshold

    # gray
    gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
    med_blur = cv2.medianBlur(gray,1)
    
    # get ersion from GUI sliders.
    erosion_size = cv2.getTrackbarPos('erosion size', FrameName)
    e_kernel = np.ones((erosion_size, erosion_size), np.uint8)
	   
    erosion_time = cv2.getTrackbarPos('erosion time', FrameName)
    erosion = cv2.erode(med_blur, e_kernel, iterations = erosion_time)
   
# get dilation from GUI sliders.
    	       
    dilation_size = cv2.getTrackbarPos('dilation size', FrameName)
    d_kernel = np.ones((dilation_size, dilation_size), np.uint8)
			       
    dilation_time = cv2.getTrackbarPos('dilation time', FrameName)
    dilation = cv2.dilate(erosion,d_kernel,iterations = dilation_time)
    
    threshold = cv2.getTrackbarPos('threshold', FrameName)
    (_, binary) = cv2.threshold(dilation, threshold, 255, cv2.THRESH_BINARY)
  
    return binary

###################################   ROI   ###############################

# create ROI track bar
def ROIbar(FrameName):
    cv2.createTrackbar('x1', FrameName, 100 , 637, nothing)
    cv2.createTrackbar('y1', FrameName, 1 , 748, nothing)
    cv2.createTrackbar('w1', FrameName, 1 , 637, nothing)
    cv2.createTrackbar('h1', FrameName, 1 , 748, nothing)
    cv2.createTrackbar('x2', FrameName, 1 , 637, nothing)
    cv2.createTrackbar('y2', FrameName, 1 , 748, nothing)
    cv2.createTrackbar('w2', FrameName, 1 , 637, nothing)
    cv2.createTrackbar('h2', FrameName, 1 , 748, nothing)

    # get ROI position
def ROI(FrameName, inputImg, outputImg):
    global x1, y1, w1, h1, x2, y2, w2, h2
    x1 = cv2.getTrackbarPos('x1',FrameName)
    y1 = cv2.getTrackbarPos('y1',FrameName)
    w1 = cv2.getTrackbarPos('w1',FrameName)
    h1 = cv2.getTrackbarPos('h1',FrameName)
    x2 = cv2.getTrackbarPos('x2',FrameName)
    y2 = cv2.getTrackbarPos('y2',FrameName)
    w2 = cv2.getTrackbarPos('w2',FrameName)
    h2 = cv2.getTrackbarPos('h2',FrameName)
    
    ROI_image = inputImg.copy()
    ROI_image = cv2.rectangle(ROI_image, (x1,y1), (x1+w1,y1+h1),[0,255,0], 2)
    ROI_image = cv2.rectangle(ROI_image, (x2,y2), (x2+w2,y2+h2),[0,0,255], 2)
    ROI_image = cv2.circle(ROI_image, (x1,y1), 6, [0,255,0], -1)
    ROI_image = cv2.circle(ROI_image, (x2,y2), 6, [0,0,255], -1)
    
    ROIfront = outputImg[y1:y1+h1, x1:x1+w1] # [y,x]
    ROIback = outputImg[y2:y2+h2, x2:x2+w2]

    areaf = np.sum(ROIfront)/255
    areab = np.sum(ROIback)/255
  
    return ROI_image, ROIfront, ROIback, areaf, areab


################################ obstacle detection ###############################

def ob_detection(front, back, front_init, back_init, thresf, thresb):
   
    if front>front_init+thresf or front<front_init-thresf:
        f = 1
    else:
        f = 0
    
    if back>back_init+thresb or back<back_init-thresb:
        b = 1
    else:
        b = 0

    return f, b
############################### save parameter ################################

def saveParameter(id):
    global parameter
    parameter = FilterParameter()
    parameter.hsv_lh = lowHue
    parameter.hsv_ls = lowSat 
    parameter.hsv_lv = lowVal
    parameter.hsv_hh = highHue
    parameter.hsv_hs = highSat
    parameter.hsv_hv = highVal
    parameter.clear_es = erosion_size
    parameter.clear_et = erosion_time
    parameter.clear_ds = dilation_size
    parameter.clear_dt = dilation_time
    parameter.clear_th = threshold
    parameter.roi_x1 = x1
    parameter.roi_y1 = y1
    parameter.roi_w1 = w1
    parameter.roi_h1 = h1
    parameter.roi_x2 = x2
    parameter.roi_y2 = y2
    parameter.roi_w2 = w2
    parameter.roi_h2 = h2
    parameter.id = id

##################################################################

# set parameter
ROIbar('ROI1')
ROIbar('ROI2')
HSVbar('color1')
HSVbar('color2')
clearerbar('color1')
clearerbar('color2')
p_f1 = []
p_b1 = []
p_f2 = []
p_b2 = []

areaf1_init, areab1_init, det_thres1 = 1680, 1330, 200
areaf2_init, areab2_init, det_thres2 = 1500, 980, 200

def GUI_parameter():
    while(True):
        ret1, frame1 = cap1.read()
	ret2, frame2 = cap2.read()
	hsv1 = HSV('color1', frame1)
	hsv2 = HSV('color2', frame2)
	thres1 = clearer('color1', hsv1)
        thres2 = clearer('color2', hsv2)
        ROI1, ROI1front, ROI1back, areaf1, areab1 = ROI('ROI1', frame1, thres1)
        ROI2, ROI2front, ROI2back, areaf2, areab2 = ROI('ROI2', frame2, thres2)
        global p_f1, p_f2, p_b1, p_b2 
        p_f1 = np.append(p_f1, areaf1)
        p_b1 = np.append(p_b1, areab1)
        p_f2 = np.append(p_f2, areaf2)
        p_b2 = np.append(p_b2, areab2)
       
        if len(p_f1) == 20:
            areaf1 = sum(p_f1)//len(p_f1)
            areab1 = sum(p_b1)//len(p_b1)
            areaf2 = sum(p_f2)//len(p_f2)
            areab2 = sum(p_b2)//len(p_b2)
            print('areaf1:%d, areab1:%d   areaf2:%d, areab2:%d' %(areaf1, areab1, areaf2, areab2))
            p_f1 = []
            p_b1 = []
            p_f2 = []
            p_b2 = []

            global areaf1_init, areab1_init, det_thres1, areaf2_init, areab2_init, det_thres2
            f1, b1 = ob_detection(areaf1, areab1, areaf1_init, areab1_init, det_thres1, det_thres1)
            f2, b2 = ob_detection(areaf2, areab2, areaf2_init, areab2_init, det_thres2, det_thres2)
 
            flag = f1 + b1 + f2 + b2
            if flag == 2:
                laser_detect = 'GO!'
            elif flag == 3:
                laser_detect = 'watch out!'
            elif flag == 4:
                laser_detect = 'STOP!'
            else:
                laser_detect = 'error: didn\'t detect flag'
            '''
	    global parameter
	    parameter = FilterParameter()
            parameter.hsv_lh = lowHue
            parameter.hsv_ls = lowSat 
	    parameter.hsv_lv = lowVal
	    parameter.hsv_hh = highHue
	    parameter.hsv_hs = highSat
	    parameter.hsv_hv = highVal
            parameter.clear_es = erosion_size
	    parameter.clear_et = erosion_time
	    parameter.clear_ds = dilation_size
	    parameter.clear_dt = dilation_time
	    parameter.clear_th = threshold
            parameter.roi_x1 = x1
	    parameter.roi_y1 = y1
	    parameter.roi_w1 = w1
	    parameter.roi_h1 = h1
	    parameter.roi_x2 = x2
	    parameter.roi_y2 = y2
	    parameter.roi_w2 = w2
	    parameter.roi_h2 = h2
            '''
            cv2.imshow('color1', thres1)
	    cv2.imshow('color2', thres2)
            cv2.imshow('ROI1', ROI1)
            cv2.imshow('ROI2', ROI2)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
	
        elif k == 49:
            areaf1_init, areab1_init = areaf1, areab1
	    hsv1 = HSV('color1', frame1)
	    thres1 = clearer('color1', hsv1)
	    ROI('ROI1', frame1, thres1)
	    saveParameter('1')
	    print('camera1 parameter......')
	    talker()
            print('camera1 saved')
        elif k == 50:
	    areaf2_init, areab2_init = areaf2, areab2
            hsv2 = HSV('color2', frame2)
            thres2 = clearer('color2', hsv2)
            ROI('ROI2', frame2, thres2)
	    saveParameter('2')
	    print('camera2 parameter......')
	    talker()
            print('camera2 saved')
        
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def talker():
    
    pub = rospy.Publisher('laser_parameter', FilterParameter, queue_size=1)
    rospy.init_node('laser_parameter', anonymous=True)
    #rate = rospy.Rate(10) # 10Hz
 
    rate = rospy.Rate(10) # 10Hz
    #cv2.namedWindow('save')
    
    while not rospy.is_shutdown():
        #if cv2.waitKey(1) & 0xFF == ord('s'):
        connections = pub.get_num_connections()
        rospy.loginfo('Connection: %d', connections)
        if connections > 0:
            pub.publish(parameter) ############ publish parameter ###############
            rospy.loginfo('Published')
            break
        rate.sleep()


if __name__=='__main__':
    try:
        GUI_parameter()
    except rospy.ROSInterruptException:
        pass
