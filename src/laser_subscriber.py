#!/usr/bin/env python
# 20191113 WT
# v1.0(to v2.1 laser_parameter.py)
# subscribe custom msg FilterParameter

import rospy
from std_msgs.msg import String
from laser_back.msg import FilterParameter


def callback(data):
    if data.id == '1':
        rospy.loginfo('this is camera 1')
	rospy.loginfo(data)
	global flag1
	flag1 = 1
    elif data.id == '2':
        rospy.loginfo('this is camera 2')
	rospy.loginfo(data)
        global flag2
	flag2 = 1
    if flag1+flag2 ==2:
        print('camera1 and camera2 saved')
	

def listener():
    
    rospy.init_node('listener', anonymous=True)
    global sub	
    sub = rospy.Subscriber("laser_parameter", FilterParameter, callback)
	
    rospy.spin()
		
if __name__ == '__main__':
    flag1 = 0
    flag2 = 0
    listener()
    print('123123123123123123')
