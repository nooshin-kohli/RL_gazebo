#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState,Imu
import time

pub = rospy.Publisher('/joint_states_new', JointState, queue_size=10)

def callback(data):
    # print('test ', data)
    time.sleep(0.009) #### 23*4
    # print('*********************')
    pub.publish(data)

if __name__ == '__main__':
    rospy.init_node("jointstate_node")
    # rospy.loginfo('data subscriber...')

    rate = rospy.Rate(50)
    rospy.Subscriber('/legs/joint_states', JointState, callback)

    while not rospy.is_shutdown():
        # rospy.loginfo('waiting...')
        rate.sleep()
    else:
        rospy.loginfo('shutdown!')