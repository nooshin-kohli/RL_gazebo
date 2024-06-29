#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState

def set_initial_joint_positions():
    rospy.init_node('set_initial_joint_positions')
    
    pub = rospy.Publisher('/legs/joint_states', JointState, queue_size=10)
    
    rate = rospy.Rate(10)  # 10 Hz
    
    joint_state = JointState()
    joint_state.name = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint','RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'\
        'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint','RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
    joint_state.position = [0.1,0.8,-1.5,  0.1,0.8,-1.5,  -0.1,0.8,-1.5,  -0.1,0.8,-1.5]
    
    while not rospy.is_shutdown():
        joint_state.header.stamp = rospy.Time.now()
        pub.publish(joint_state)
        rate.sleep()

if __name__ == '__main__':
    try:
        set_initial_joint_positions()
    except rospy.ROSInterruptException:
        pass
