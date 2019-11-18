#!/usr/bin/env python
#!coding=utf-8

'''
20191118 WT
v2.1(to v1 cam_pub1.py and v1 cam_pub2.py)
subscribe image from /laserimage1 and /laserimage2 topics and set parameter using GUI
save parameter as ymal file
'''

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time


class setParameter():
    def __init__(self):
        rospy.init_node('display_laserimage', anonymous=True)
        
        global count1, count2
        count1 = 0
	count2 = 0
                
	self.bridge1 = CvBridge()
	self.bridge2 = CvBridge()
        self.cam1_img = cv2.imread('/home/jjting/laser/output.jpg') # load random image to initialize
        self.cam2_img = cv2.imread('/home/jjting/laser/output.jpg')

	rospy.Subscriber('/laserimage1', Image, self.callback1)
        rospy.Subscriber('/laserimage2', Image, self.callback2)

        cv2.namedWindow('color1')
        cv2.namedWindow('color2')
        cv2.namedWindow('ROI1')
        cv2.namedWindow('ROI2')

        self.ROIbar('ROI1')
	self.ROIbar('ROI2')
	self.HSVbar('color1')
	self.HSVbar('color2')
        self.clearerbar('color1')
	self.clearerbar('color2')
        p_f1 = []
	p_b1 = []
	p_f2 = []
	p_b2 = []
        areaf1_init, areab1_init, det_thres1 = 1680, 1330, 200
        areaf2_init, areab2_init, det_thres2 = 1500, 980, 200
        while not rospy.is_shutdown():
	    cv2.imshow("cam1 frame", self.cam1_img)
            cv2.imshow("cam2 frame", self.cam2_img)
	   
            hsv1 = self.HSV('color1', self.cam1_img)
            hsv2 = self.HSV('color2', self.cam2_img)
            thres1 = self.clearer('color1', hsv1)
	    thres2 = self.clearer('color2', hsv2)
            ROI1, ROI1front, ROI1back, areaf1, areab1 = self.ROI('ROI1', self.cam1_img, thres1)
            ROI2, ROI2front, ROI2back, areaf2, areab2 = self.ROI('ROI2', self.cam2_img, thres2)

            p_f1 = np.append(p_f1, areaf1)
            p_f2 = np.append(p_f2, areaf2)
            p_b1 = np.append(p_b1, areab1)
            p_b2 = np.append(p_b2, areab2)

            if len(p_f1) == 20:
	        areaf1 = sum(p_f1)//len(p_f1)
	        areaf2 = sum(p_f2)//len(p_f2)
                areab1 = sum(p_b1)//len(p_b1)
                areab2 = sum(p_b2)//len(p_b2)
		print('areaf1:%d, areab1:%d   areaf2:%d, areab2:%d' %(areaf1, areab1, areaf2, areab2))
                p_f1 = []
                p_b1 = []
        	p_f2 = []
        	p_b2 = []
                
		f1, b1 = self.ob_detection(areaf1, areab1, areaf1_init, areab1_init, det_thres1, det_thres1)
	        f2, b2 = self.ob_detection(areaf2, areab2, areaf2_init, areab2_init, det_thres2, det_thres2)

		flag = f1 + b1 + f2 + b2
	        if flag == 2:
	            self.laser_detect = 'GO!'
	        elif flag == 3:
	            self.laser_detect = 'watch out!'
	        elif flag == 4:
	            self.laser_detect = 'STOP!'
	        else:
	            self.laser_detect = 'error: didn\'t detect flag'
            
	        print(self.laser_detect)

	    k = cv2.waitKey(3) & 0xFF
	    if k == 27:
	        print('Quit subscriber......')
		path = '/home/jjting/laser/test.txt'
                file = open(path, 'w')#########################
		print('parameter saved to: %s' % path)

	        break
	    

##############################   subscribe Img   ########################
    def callback1(self, data):
    
        global count1
        count1 = count1 + 1

        if count1 == 1:
            count1 = 0
            self.cam1_img = self.bridge1.imgmsg_to_cv2(data, "bgr8")
            
	    #print('img1 received')
        else:
            pass

    def callback2(self, data):
    
        global count2, re2
        count2 = count2 + 1

        if count2 == 1:
            count2 = 0
            self.cam2_img = self.bridge2.imgmsg_to_cv2(data, "bgr8")
            
            #print('img2 received')
        else:
            pass

#############################   HSV   ##########################
    def HSVbar(self, FrameName):
        
        # Lower range colour sliders.
        cv2.createTrackbar('lowHue', FrameName, 1, 255, self.nothing)
        cv2.createTrackbar('lowSat', FrameName, 1, 255, self.nothing)
        cv2.createTrackbar('lowVal', FrameName, 1, 255, self.nothing)
        # Higher range colour sliders.
        cv2.createTrackbar('highHue', FrameName, 255, 255, self.nothing)
        cv2.createTrackbar('highSat', FrameName, 255, 255, self.nothing)
        cv2.createTrackbar('highVal', FrameName, 255, 255, self.nothing)
			
# Get HSV values from the GUI sliders.
    def HSV(self, FrameName, inputimg):
        #global lowHue, lowSat, lowVal, highHue, highSat, highVal
     
        self.lowHue = cv2.getTrackbarPos('lowHue', FrameName)
        self.lowSat = cv2.getTrackbarPos('lowSat', FrameName)
        self.lowVal = cv2.getTrackbarPos('lowVal', FrameName)
        self.highHue = cv2.getTrackbarPos('highHue', FrameName)
        self.highSat = cv2.getTrackbarPos('highSat', FrameName)
        self.highVal = cv2.getTrackbarPos('highVal', FrameName)
        hsv = cv2.cvtColor(inputimg, cv2.COLOR_BGR2HSV)
				 
        lower_red = np.array([self.lowHue, self.lowSat, self.lowVal])
        upper_red = np.array([self.highHue, self.highSat, self.highVal])
		                 
        mask = cv2.inRange(hsv, lower_red, upper_red)
				                         
        res = cv2.bitwise_and(inputimg, inputimg, mask=mask)
										     
        return res 

###########################   clear   ###########################

    def clearerbar(self, FrameName):
        cv2.createTrackbar('erosion time', FrameName, 1, 6, self.nothing)
        cv2.createTrackbar('erosion size', FrameName, 1, 4, self.nothing)
        cv2.createTrackbar('dilation time', FrameName, 1, 6, self.nothing)
        cv2.createTrackbar('dilation size', FrameName, 1, 4, self.nothing)
        cv2.createTrackbar('threshold', FrameName, 1, 255, self.nothing)
		     
    def clearer(self, FrameName, inputimg):
        global erosion_size, erosion_time, dilation_size, dilation_time, threshold
			 
        # gray
        gray = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
        med_blur = cv2.medianBlur(gray, 1)
         
        # get ersion from GUI sliders.
        self.erosion_size = cv2.getTrackbarPos('erosion size', FrameName)
        e_kernel = np.ones((self.erosion_size, self.erosion_size), np.uint8)
						                
        self.erosion_time = cv2.getTrackbarPos('erosion time', FrameName)
        erosion = cv2.erode(med_blur, e_kernel, iterations = self.erosion_time)
							   
        # get dilation from GUI sliders.						   
        self.dilation_size = cv2.getTrackbarPos('dilation size', FrameName)
        d_kernel = np.ones((self.dilation_size, self.dilation_size), np.uint8)
									   
        self.dilation_time = cv2.getTrackbarPos('dilation time', FrameName)
        dilation = cv2.dilate(erosion,d_kernel,iterations = self.dilation_time)
											   
        self.threshold = cv2.getTrackbarPos('threshold', FrameName)
        (_, binary) = cv2.threshold(dilation, self.threshold, 255, cv2.THRESH_BINARY)
									   
        return binary
###################################   ROI   ###############################
    def nothing(self, x):
        pass

    # create ROI track bar
    def ROIbar(self, FrameName):
        cv2.createTrackbar('x1', FrameName, 100 , 637, self.nothing)
        cv2.createTrackbar('y1', FrameName, 1 , 748, self.nothing)
        cv2.createTrackbar('w1', FrameName, 1 , 637, self.nothing)
        cv2.createTrackbar('h1', FrameName, 1 , 748, self.nothing)
        cv2.createTrackbar('x2', FrameName, 1 , 637, self.nothing)
        cv2.createTrackbar('y2', FrameName, 1 , 748, self.nothing)
        cv2.createTrackbar('w2', FrameName, 1 , 637, self.nothing)
        cv2.createTrackbar('h2', FrameName, 1 , 748, self.nothing)
				
    # get ROI position
    def ROI(self, FrameName, inputImg, outputImg):
        #global x1, y1, w1, h1, x2, y2, w2, h2
        self.x1 = cv2.getTrackbarPos('x1', FrameName)
        self.y1 = cv2.getTrackbarPos('y1', FrameName)
        self.w1 = cv2.getTrackbarPos('w1', FrameName)
        self.h1 = cv2.getTrackbarPos('h1', FrameName)
        self.x2 = cv2.getTrackbarPos('x2', FrameName)
        self.y2 = cv2.getTrackbarPos('y2', FrameName)
        self.w2 = cv2.getTrackbarPos('w2', FrameName)
        self.h2 = cv2.getTrackbarPos('h2', FrameName)
  									
        ROI_image = inputImg.copy()
        ROI_image = cv2.rectangle(ROI_image, (self.x1, self.y1), (self.x1+self.w1, self.y1+self.h1),[0,255,0], 2)
        ROI_image = cv2.rectangle(ROI_image, (self.x2, self.y2), (self.x2+self.w2, self.y2+self.h2),[0,0,255], 2)
        ROI_image = cv2.circle(ROI_image, (self.x1, self.y1), 6, [0,255,0], -1)
        ROI_image = cv2.circle(ROI_image, (self.x2, self.y2), 6, [0,0,255], -1)
  									    
        ROIfront = outputImg[self.y1:self.y1+self.h1, self.x1:self.x1+self.w1] # [y,x]
        ROIback = outputImg[self.y2:self.y2+self.h2, self.x2:self.x2+self.w2]

        areaf = np.sum(ROIfront)/255
        areab = np.sum(ROIback)/255
  
        return ROI_image, ROIfront, ROIback, areaf, areab
    
################################ obstacle detection ###############################

    def ob_detection(self, front, back, front_init, back_init, thresf, thresb):

        if front>front_init+thresf or front<front_init-thresf:
            f = 1
        else:
            f = 0
			
        if back>back_init+thresb or back<back_init-thresb:
            b = 1
        else:
            b = 0
						
        return f, b
											
####################################    main    #################################		
if __name__ == '__main__':
    setParameter()
    
    
