#!/usr/bin/env python
#!coding=utf-8

'''
20191114 WT
v1.0
test file to subscribe image from /laserimage topic
'''

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def callback(data):
    
    global count, bridge
    count = count + 1
    if count == 1:
        count = 0
        cam_img = bridge.imgmsg_to_cv2(data, "bgr8")
	cv2.imshow("cam1 frame", cam_img)
	#print('img received')
	cv2.waitKey(1)
    else:
        pass

def displayImage():
    rospy.init_node('display_laserimage', anonymous=True)

    global count, bridge
    count = 0
    bridge = CvBridge()

    rospy.Subscriber('/laserimage2', Image, callback)
    rospy.spin()

if __name__ == '__main__':
    displayImage()
