#!/usr/bin/env python
#!coding=utf-8

'''
20191115 WT
v1.0(to v1 cam_pub1.py and v1 cam_pub2.py)
subscribe image from /laserimage1 and /laserimage2 topics and set parameter using GUI
save parameter as ymal file
'''

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class setParameter():
    def __init__(self):
        rospy.init_node('display_laserimage', anonymous=True)
        
        global count1, count2
        count1 = 0
	count2 = 0
        self.bridge1 = CvBridge()
	self.bridge2 = CvBridge()

        rospy.Subscriber('/laserimage1', Image, self.callback1)
        rospy.Subscriber('/laserimage2', Image, self.callback2)

       


    def callback1(self, data):
    
        global count1
        count1 = count1 + 1

        if count1 == 1:
            count1 = 0
            self.cam1_img = self.bridge1.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("cam1 frame", self.cam1_img)
	    print('img1 received')
            cv2.waitKey(3)
	    
        else:
            pass

    def callback2(self, data):
    
        global count2
        count2 = count2 + 1

        if count2 == 1:
            count2 = 0
            self.cam2_img = self.bridge2.imgmsg_to_cv2(data, "bgr8")
            #cv2.imshow("cam2 frame", self.cam2_img)
	    print('img2 received')
            #cv2.waitKey(3)
	    
        else:
            pass


if __name__ == '__main__':
    setParameter()
    rospy.spin()
