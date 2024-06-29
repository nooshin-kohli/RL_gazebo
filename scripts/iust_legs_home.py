#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 19:54:40 2022
@author: Nooshin Kohli
"""

from turtle import shape
import numpy as np
# from quadruped_Class import quadruped_class
# from quadControl_Class import Quadruped_ControlClass, Control
from lib2to3.pytree import type_repr
import numpy as np
import rospy
import matplotlib.pyplot as plt
from numpy import ndarray
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import JointState,Imu
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
# from robot_class import ROBOT
import os
import time
import lcm
from obslcm import observ_t, action_t
from gazebo_msgs.msg import LinkStates

########################################################## defining publishers ##########################################################

FL_hip = rospy.Publisher('/legs/FL_hip_joint_effort_controller/command', Float64, queue_size=10)
FL_thigh = rospy.Publisher('/legs/FL_thigh_joint_effort_controller/command', Float64, queue_size=10)
FL_calf = rospy.Publisher('/legs/FL_calf_joint_effort_controller/command', Float64, queue_size=10)
FR_hip = rospy.Publisher('/legs/FR_hip_joint_effort_controller/command', Float64, queue_size=10)
FR_thigh = rospy.Publisher('/legs/FR_thigh_joint_effort_controller/command', Float64, queue_size=10)
FR_calf = rospy.Publisher('/legs/FR_calf_joint_effort_controller/command', Float64, queue_size=10)
RL_hip = rospy.Publisher('/legs/RL_hip_joint_effort_controller/command', Float64, queue_size=10)
RL_thigh = rospy.Publisher('/legs/RL_thigh_joint_effort_controller/command', Float64, queue_size=10)
RL_calf = rospy.Publisher('/legs/RL_calf_joint_effort_controller/command', Float64, queue_size=10)
RR_hip = rospy.Publisher('/legs/RR_hip_joint_effort_controller/command', Float64, queue_size=10)
RR_thigh = rospy.Publisher('/legs/RR_thigh_joint_effort_controller/command', Float64, queue_size=10)
RR_calf = rospy.Publisher('/legs/RR_calf_joint_effort_controller/command', Float64, queue_size=10)
rospy.init_node('legs_homing', anonymous=True)

###################################################### defining variables ######################################################

#### inital values for joints 
obs = observ_t()
q = np.zeros(12) 
qdot = np.zeros(12) 
q_d = np.zeros(12)
qdot_d = np.zeros(12)
kpp = 1
count = 30
base_pos = [0,0,0]
base_vel = [0,0,0]
pre_base_pos = [0,0,0]
base_qdot = np.zeros(12) 
angular = [0,0,0]
pre_action = np.zeros(12) 
base_angular = np.zeros(3)
lc = lcm.LCM("udpm://224.0.55.55:5001?ttl=225") #//224.0.55.55:5001?ttl=225"
t_pre = time.time()


tpre = rospy.get_time()
q_rbdl = np.zeros(12)
qdot_rbdl = np.zeros(12)
kp = 10*np.identity(12)

kd = 0.5*np.identity(12)


t_first = time.time()

def pubish(effort):
    """
    This function publishes calculated tau to joints.
    """

    RL_hip.publish(effort[7])
    RL_thigh.publish(effort[8])
    RL_calf.publish(effort[6])
    RR_hip.publish(effort[10])
    RR_thigh.publish(effort[11])
    RR_calf.publish(effort[9])
    FL_hip.publish(effort[1])
    FL_thigh.publish(effort[2])
    FL_calf.publish(effort[0])
    FR_hip.publish(effort[4])
    FR_thigh.publish(effort[5])
    FR_calf.publish(effort[3])



###################################################### desired and task values ######################################################

#### inital values for joints  
q_d = np.zeros(12)
# q_d[0] = -1.5
# q_d[1] = -0.1
# q_d[2] = 0.8

# q_d[3] = -1.5
# q_d[4] = -0.1
# q_d[5] = 0.8

# q_d[6] = -1.5
# q_d[7] = 0.1
# q_d[8] = 0.8

# q_d[9] = -1.5
# q_d[10] = 0.1
# q_d[11] = 0.8
qdot_d = np.zeros(12)
kpp = 10
kp = kpp*np.identity(12)
count = 30
t_first = time.time()
t_pre = time.time()
def callback(data):
    global kp,kd,tau,q_d, qdot_d, kpp,count, t_first, t_pre

    ################################################ give ros data to rbdl ######################################
    q = data.position
    q = np.asarray(q)
    # print("q")
    # print(q)
    qdot = data.velocity
    qdot = np.asarray(qdot)
    
    # q_rbdl,qdot_rbdl = ros_to_rbdl(q,qdot)
    # print(time.time()-t_pre)


    if time.time()-t_first<=10:
        q_d=q.copy()
        q_d[0] = -1.5
        q_d[3] = -1.5
        q_d[6] = -1.5
        q_d[9] = -1.5
        # print(q_d)
        # print(q)
        error = q_d - q
        print("homing calf")
        tau_pid = np.dot(error, kp) + np.dot(qdot_d-qdot, kd)
        
    elif time.time()-t_first>10 and time.time()-t_first<20:
        q_d=q.copy()
        q_d[2] = 0.8
        q_d[5] = 0.8
        q_d[8] = 0.8
        q_d[11] = 0.8
        q_d[0] = -1.5
        q_d[3] = -1.5
        q_d[6] = -1.5
        q_d[9] = -1.5
        error = q_d - q
        print("homing thigh")
        tau_pid = np.dot(error, kp) + np.dot(qdot_d-qdot, kd)
    elif time.time()-t_first>=20 and time.time()-t_first<=30:
        q_d = q.copy()
        kp = 15*np.identity(12)
        q_d[1] = -0.1
        q_d[4] = 0.1
        q_d[7] = -0.1
        q_d[10] = 0.1
        q_d[2] = 0.8
        q_d[5] = 0.8
        q_d[8] = 0.8
        q_d[11] = 0.8
        q_d[0] = -1.5
        q_d[3] = -1.5
        q_d[6] = -1.5
        q_d[9] = -1.5
        error = q_d - q
        print("homing hip")
        tau_pid = np.dot(error, kp) + np.dot(qdot_d-qdot, kd)
        
    
    # error = q_d - q
    # error_dot = qdot_d - qdot
    # if kpp<20:
    #     if count>0:
    #         
    #         count = count-1
    #     else:
    #         kpp = kpp+1
    #         count = 30
    # else:
    #     kp = kpp*np.identity(12)
        # print(q)
    # print(kpp)
    # print(q_d)
    # print(error)
    

    

    if (time.time()-t_first<30):
        pubish(tau_pid)
    else:
        rospy.signal_shutdown("reason")

    # t_vec.appen(time.time())
    # print(tau_pid)
    t_pre = time.time()



q_minicheetah = []
def joints(q):
    global q_minicheetah
    q_minicheetah = []
    q_minicheetah.append(q[4])
    q_minicheetah.append(q[5])
    q_minicheetah.append(q[3])

    q_minicheetah.append(q[1])
    q_minicheetah.append(q[2])
    q_minicheetah.append(q[0])

    q_minicheetah.append(q[10])
    q_minicheetah.append(q[11])
    q_minicheetah.append(q[9])

    q_minicheetah.append(q[7])
    q_minicheetah.append(q[8])
    q_minicheetah.append(q[6])

    return q_minicheetah





def main():
    # rospy.Subscriber("/linkstatedata", LinkStates, base_state)
    # rospy.init_node("Node")
    # if (time.time()-t_first)>20:
    # rospy.Subscriber("/joint_states_new", JointState, callback)
    rospy.Subscriber("/legs/joint_states", JointState, callback)
    rospy.spin()
    if rospy.is_shutdown():
        os.system('python observation_node.py')
        # plt.figure()
        # plt.polot(t_vec,error)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInternalException:
        pass