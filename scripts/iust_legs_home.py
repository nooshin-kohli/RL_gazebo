#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 19:54:40 2022
@author: Nooshin Kohli
"""

from turtle import shape
import numpy as np
from quadruped_Class import quadruped_class
from quadControl_Class import Quadruped_ControlClass, Control
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
import time
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
rospy.init_node('legs', anonymous=True)

###################################################### defining variables ######################################################

tpre = rospy.get_time()
q_rbdl = np.zeros(12)
qdot_rbdl = np.zeros(12)
kp = 10*np.identity(12)

kd = 0.1*np.identity(12)




def pubish(effort):
    """
    This function publishes calculated tau to joints.
    """
    RL_hip.publish(effort[6])
    RL_thigh.publish(effort[7])
    RL_calf.publish(effort[8])
    RR_hip.publish(effort[9])
    RR_thigh.publish(effort[10])
    RR_calf.publish(effort[11])
    FL_hip.publish(effort[0])
    FL_thigh.publish(effort[1])
    FL_calf.publish(effort[2])
    FR_hip.publish(effort[3])
    FR_thigh.publish(effort[4])
    FR_calf.publish(effort[5])
    

def ros_to_rbdl(q,qdot):
    global q_rbdl,qdot_rbdl
    """
    This function corrects the sequence of q and adot 
    """
    # FL:
    q_rbdl[0] = q[1]
    q_rbdl[1] = q[2]
    q_rbdl[2] = q[0]
    # FR:
    q_rbdl[3] = q[4]
    q_rbdl[4] = q[5]
    q_rbdl[5] = q[3]
    # RL:
    q_rbdl[6] = q[7]
    q_rbdl[7] = q[8]
    q_rbdl[8] = q[6]
    # RR:
    q_rbdl[9] = q[10]
    q_rbdl[10] = q[11]
    q_rbdl[11] = q[9]
    # FL:
    qdot_rbdl[0] = qdot[1]
    qdot_rbdl[1] = qdot[2]
    qdot_rbdl[2] = qdot[0]
    # FR:
    qdot_rbdl[3] = qdot[4]
    qdot_rbdl[4] = qdot[5]
    qdot_rbdl[5] = qdot[3]
    # RL:
    qdot_rbdl[6] = qdot[7]
    qdot_rbdl[7] = qdot[8]
    qdot_rbdl[8] = qdot[6]
    # RR:
    qdot_rbdl[9] = qdot[10]
    qdot_rbdl[10] = qdot[11]
    qdot_rbdl[11] = qdot[9]
    return q_rbdl,qdot_rbdl

###################################################### desired and task values ######################################################

#### inital values for joints  
q_d = np.zeros(12)
qdot_d = np.zeros(12)
kpp = 1
count = 30
def callback(data):
    global kp,kd,tau,q_d, qdot_d, kpp,count
  
    ################################################ give ros data to rbdl ######################################
    q = data.position
    q = np.asarray(q)
    qdot = data.velocity
    qdot = np.asarray(qdot)
    q_rbdl,qdot_rbdl = ros_to_rbdl(q,qdot)


    error = q_d - q_rbdl
    error_dot = qdot_d - qdot_rbdl
    if kpp<21:
        if count>0:
            kp = kpp*np.identity(12)
            count = count-1
        else:
            kpp = kpp+1
            count = 30
    else:
        kp = kpp*np.identity(12)
    print(kpp)
    tau_pid = np.dot(error, kp) + np.dot(error_dot, kd)
    pubish(tau_pid)
    t_vec.appen(time.time())
    # print(tau_pid)
    

def odom_data(data):
    global pose_com
    # print("odom data ======================================")
    pose_com = np.zeros(3)
    pose_com[0] = data.pose.pose.position.x
    pose_com[1] = data.pose.pose.position.y
    pose_com[2] = data.pose.pose.position.z
    # print(pose_com)

def imu_data(data):
    global orientation_com
    orientation_com[0] = data.orientation.x
    orientation_com[1] = data.orientation.y
    orientation_com[2] = data.orientation.z
    # print(orientation_com)
    # print("imu data +++++++++++++++++++++++++++++++++++++++")

    # print(data)

        


def main():
    # rospy.Subscriber("/odom", Odometry, odom_data)
    # rospy.Subscriber("/imu/data", Imu, imu_data)
    # rospy.init_node("Node")
    rospy.Subscriber("/legs/joint_states", JointState, callback)
    rospy.spin()
    if rospy.is_shutdown():
        plt.figure()
        plt.polot(t_vec,error)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInternalException:
        pass