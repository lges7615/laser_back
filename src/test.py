#!/usr/bin/env python
# test only

import rospy
from std_msgs.msg import String

def testpub():
    rospy.init_node('test', anonymous=True)
    pub = rospy.Publisher('/test', String, queue_size=1)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        print('into the loop')
	exist_, _, _ = get_published_topics(namespace='/test')
        print(exist)
	pub.publish('go go go ')
	print('published')
        rate.sleep()
       
if __name__ == '__main__':
    testpub()
    
