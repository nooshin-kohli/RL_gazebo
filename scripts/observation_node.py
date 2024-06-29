#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 19:54:40 2022
@author: Nooshin Kohli
"""

from turtle import shape
import numpy as np
from lib2to3.pytree import type_repr
import numpy as np
import rospy
# import matplotlib.pyplot as plt
from numpy import ndarray
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import JointState,Imu
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import LinkStates
# from robot_class import ROBOT
from scipy.signal import butter, lfilter, freqz
import time
import lcm
from obslcm import observ_t, action_t
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
rospy.init_node('obsevrvation_node', anonymous=True)

###################################################### defining variables ######################################################

obs = observ_t()
# obs = np.zeros(45)
orientation_com = [0,0,0]
tpre = rospy.get_time()
q_rbdl = np.zeros(12)
qdot_rbdl = np.zeros(12)
kp = 10*np.identity(12)
kd = 0.1*np.identity(12)


def pubish(effort):
    """
    This function publishes calculated tau to joints.
    """
    RL_hip.publish(effort[9])
    RL_thigh.publish(effort[10])
    RL_calf.publish(effort[11])
    RR_hip.publish(effort[6])
    RR_thigh.publish(effort[7])
    RR_calf.publish(effort[8])
    FL_hip.publish(effort[3])
    FL_thigh.publish(effort[4])
    FL_calf.publish(effort[5])
    FR_hip.publish(effort[0])
    FR_thigh.publish(effort[1])
    FR_calf.publish(effort[2])
    

def ros_to_rbdl(q,qdot,action):
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


###################################################### lowpass filter ######################################################

# Filter parameters
order = 6
fs = 30.0       # rate, Hz
cutoff = 3.667  # Hz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(obs_noisy, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, obs_noisy)
    return y

###################################################### desired and task values ######################################################

#### inital values for joints 
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
qdot_list = []
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

# t_pre = time.time()
def callback(data):
    global kp,kd,tau,q_d, qdot_d, kpp,count,obs, q, qdot, pre_action, qdot_list,cutoff, fs, order
    # time.time()-t_pre
    ################################################ give ros data to rbdl ######################################
    q = data.position
    q = joints(q)
    qdot = data.velocity      
    qdot = joints(qdot)
    pre_action = data.effort
    pre_action = joints(pre_action)

    qdot = np.asarray(qdot)
    q = np.asarray(q)
    pre_action = np.asarray(pre_action)
    # print(q)
    
    obs.base_lin_vel = base_vel
    obs.base_ang_vel = base_angular
    obs.gravity = np.array([0,0,-9.8])
    obs.commands = np.array([0.0, 0, 0])
    obs.dof_pos = q
    # obs.dof_vel = qdot
    obs.action = pre_action

    obs.dof_vel = butter_lowpass_filter(qdot, cutoff, fs, order)


    # print(obs.quaternion)
    # print(obs.base_ang_vel)
    # print(obs.base_lin_vel)
    # print(obs.action)
    # print(obs.dof_pos)
    # print(obs.dof_vel)
    # print(obs.dof_pos)
    
    lc.publish("OBSERVATION", obs.encode())
    lc.handle()


def imu_data(data):
    global obs
    # obs.quaternion = np.array([data.orientation.x,data.orientation.y,data.orientation.z,data.orientation.w])
    # print(data)

def base_state(data):
    global base_pos, base_vel,t_pre,obs, base_angular,lc
    base_pos[0] = data.pose[1].position.x
    base_pos[1] = data.pose[1].position.y
    base_pos[2] = data.pose[1].position.z
    dt =  time.time()-t_pre
    
    base_vel[0] = data.twist[1].linear.x
    base_vel[1] = data.twist[1].linear.y
    base_vel[2] = data.twist[1].linear.z
    pre_base_pos[0] = data.pose[1].position.x
    pre_base_pos[1] = data.pose[1].position.y
    pre_base_pos[2] = data.pose[1].position.z
    base_angular[0] = data.twist[1].angular.x
    base_angular[1] = data.twist[1].angular.y
    base_angular[2] = data.twist[1].angular.z

    obs.quaternion = np.array([data.pose[1].orientation.x, data.pose[1].orientation.y, data.pose[1].orientation.z, data.pose[1].orientation.w])
    # obs.base_pos = np.array(base_pos)

    # obs[3:6] = base_angular
    # obs[:3] = base_vel
    # obs[6:9] = [0,0,-9.8] #gravity
    # obs[9:12] = [0, 0, 0] #commands
    
    t_pre = time.time()

    # print(obs) 
    # print(data.pose[1].position.x)
    # print("---")

t_action = time.time()
def my_handler(channel, data):
    global t_action
    msg_action = action_t.decode(data)
    # print(msg_action.tau)
    pubish(msg_action.tau)
    print(time.time()-t_action)
    t_action = time.time()
    # tau = joints(msg_action.tau)

    
def main():
    # rospy.Subscriber("/odom", Odometry, odom_data)
    # rospy.Subscriber("/imudata", Imu, imu_data)
    subscription = lc.subscribe("ACTION", my_handler)
    rospy.Subscriber("/gazebo/link_states", LinkStates, base_state)
    # rospy.init_node("Node")
    rospy.Subscriber("/legs/joint_states", JointState, callback)
    rospy.spin()
    
if __name__ == '__main__':
    try:
        main()
        # lc.handle()
    except rospy.ROSInternalException:
        pass