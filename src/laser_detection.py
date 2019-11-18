#!/usr/bin/env python
# 20191111 WT
# v2.1
# v1.0 is named camera_test.py
# dual camera with different exposure time
# HSV filter, erison and dilation
# GO, Watch out, Stop, error

#############   comment   #############
# 1. need to use usb_cam node to subscribe image for dual camera

#######################################

import rospy
import cv2
import numpy as np
#from enum import Enum
from std_msgs.msg import String
from laser_back.msg import FilterParameter
        
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap1.set(cv2.CAP_PROP_EXPOSURE,0.1)

cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap2.set(cv2.CAP_PROP_EXPOSURE,0.05)

##########################   HSV   #############################

def HSV(lowH, lowS, lowV, highH, highS, highV, inputimg):

    hsv = cv2.cvtColor(inputimg, cv2.COLOR_BGR2HSV)

    lower_red = np.array([lowH, lowS, lowV])
    upper_red = np.array([highH,highS,highV])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    res = cv2.bitwise_and(inputimg, inputimg, mask=mask)
    
    return res

########################  threshold   #########################

def threshold(es, et, ds, dt, thres, inputimg):
    # gray
    gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
    med_blur = cv2.medianBlur(gray,1)
     
    # get ersion from GUI sliders.
    e_kernel = np.ones((es, es), np.uint8) 
    erosion = cv2.erode(med_blur, e_kernel, iterations = et)
 
    # get dilation from GUI sliders.

    d_kernel = np.ones((ds, ds), np.uint8)
    dilation = cv2.dilate(erosion,d_kernel,iterations = dt)
 
    (_, binary) = cv2.threshold(dilation, thres, 255, cv2.THRESH_BINARY)
 
    return binary

#############################  ROI  ###########################

def ROI(x1,y1,w1,h1,x2,y2,w2,h2,inputimg):
    
    ROIfront = inputimg[y1:y1+h1, x1:x1+w1] # [y,x]
    ROIback = inputimg[y2:y2+h2, x2:x2+w2]
    areaf = np.sum(ROIfront)/255
    areab = np.sum(ROIback)/255
 
    return areaf, areab
###################### obstacle detection ########################

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



##################################################################

# set parameter
p_f1 = []
p_b1 = []
p_f2 = []
p_b2 = []

areaf1_init, areab1_init, det_thres1 = 1680, 1330, 200
areaf2_init, areab2_init, det_thres2 = 1500, 980, 200

def callback(data):
    if data.id == '1':
        cam1Para = data
	print('parameter 1 saved')
    elif data.id == '2':
        cam2Para = data
	print('parameter 2 saved')
        
    if flag1 + flag2 == 2:
        print('cam1 and cam2 saved')
	pub = rospy.Publisher('laser_detection', String, queue_size=1)
	rospy.init_node('laser_detection', anonymous=True)
	rate= rospy.Rate(1) # 10Hz
        
	cap1 = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(1)

        cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap1.set(cv2.CAP_PROP_EXPOSURE,0.1)

        cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap2.set(cv2.CAP_PROP_EXPOSURE,0.05)

        p_f1 = []
	p_b1 = []
	p_f2 = []
	p_b2 = []

	det_thres1 = 200
	det_thres2 = 200

	while not rospy.is_shutdown():
	    
	    _, frame1 = cap1.read() 
            _, frame2 = cap2.read()
	    hav1 = HSV(cam1Para.lowH, cam1Para.lowS, cam1Para.lowV, cam1Para.highH, cam1Para.highS, cam1Paaa.highV, frame1)
            hsv2 = HSV(cam2Para.lowH, cam2Para.lowS, cam2Para.lowV, cam2Para.highH, cam2Para.highS, cam2Para.highV, frame2)
            thres1 = threshold(cam1Para.es, cam1Para.et, cam1Para.ds, cam1Para.dt, cam1Para.th, hsv1)
            thres2 = threshold(cam2Para.es, cam2Para.et, cam2Para.ds, cam2Para.dt, cam2Para.th, hsv2)

            areaf1, areab1 = ROI('ROI1', frame1, thres1)
            areaf2, areab2 = ROI('ROI2', frame2, thres2)
                
	    #global p_f1, p_f2, p_b1, p_b2 
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
                    
	        #global areaf1_init, areab1_init, det_thres1, areaf2_init, areab2_init, det_thres2
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
                    
	        pub.publish(laser_detect)
	        rate.sleep()


def listener():
    
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('laser_parameter', FilterParameter, callback)
    rospy.spin()




'''





def talker():
    pub = rospy.Publisher('laser_detection', String, queue_size=1)
    rospy.init_node('laser_detection', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        #hello_str = "hello world %s" % rospy.get_time()
        #rospy.loginfo(hello_str)
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

                rospy.loginfo(laser_detect)
                pub.publish(laser_detect)
                rate.sleep()
	     
            

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        elif k == 49:
            areaf1_init, areab1_init = areaf1, areab1
            print('camera1 saved')
        elif k == 50:
            areaf2_init, areab2_init = areaf2, areab2
            print('camera2 saved') 
		    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
'''


if __name__=='__main__':
    try:
        flag1 = 0
	flag2 = 0
	listener()
	print('1234567879568456312451324')
    except rospy.ROSInterruptException:
        pass

