#!/usr/bin/env python
#!coding=utf-8

'''
20191115 WT
v1.0
publish camera2 image to topic named /laserimage2
'''

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import sys

def ImagePub():
    rospy.init_node('cam_laserimage2', anonymous=True)
    img_pub = rospy.Publisher('/laserimage2', Image, queue_size=1)
    rate = rospy.Rate(30) # 30Hz

    cap = cv2.VideoCapture(1)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    bridge = CvBridge()

    if not cap.isOpened():
        sys.stdout.write("cam2 is not available !")
	return -1
    
    count = 0
    while not rospy.is_shutdown():
        ret, frame = cap.read()
	#print(fps)
	if ret == 1:
	    count = count + 1
	elif ret1 != 1:
	    rospy.loginfo("cam2 failed capturing image!")
	        
	if count == 2:
            count = 0
	    msg1 = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_pub.publish(msg1)
	    #print('cam2 image frame published')
	
	rate.sleep()

if __name__ == '__main__':
    try:
        ImagePub()
    except rospy.ROSInterruptException:
        pass
	   


