#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import LinkStates
import time

pub = rospy.Publisher('/linkstatedata', LinkStates, queue_size=10)

def imu_data(data):
    # print('test ', data)
    time.sleep(0.009) #### 23*4
    # print('*********************')
    pub.publish(data)

if __name__ == '__main__':
    rospy.init_node("linkstate_node")
    # rospy.loginfo('data subscriber...')

    rate = rospy.Rate(50)
    rospy.Subscriber('/gazebo/link_states', LinkStates, imu_data)

    while not rospy.is_shutdown():
        # rospy.loginfo('waiting...')
        rate.sleep()
    else:
        rospy.loginfo('shutdown!')