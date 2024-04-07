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
rospy.init_node('command_legs', anonymous=True)

###################################################### defining variables ######################################################
Jc_cond = []
t_vec = []
tpre = rospy.get_time()
q_rbdl = np.zeros(12)
qdot_rbdl = np.zeros(12)
kp = 30*np.identity(18)
# kp = [[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0],
#       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]]
kd = 5*np.identity(18)
tau = np.zeros(12)


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
a = np.random.rand(18,18)
#### inital values for joints  
q_d = np.zeros(12)
q_d[1] = 0.6
q_d[2]= -1.02

q_d[4]= 0.6
q_d[5]= -1.02

q_d[7]= 0.6
q_d[8]= -1.02

q_d[10]= 0.6
q_d[11]= -1.02
qdot_d = np.zeros(12)

#### initail values for trunk 
pose_com = np.zeros(3)
orientation_com = np.zeros(3)
q_d_com = np.hstack((q_d, np.zeros(6)))
qdot_d_com = np.hstack((qdot_d, np.zeros(6)))


t = np.array([0])
dt = .005 # step size
p = [[ ]] # the contact feet
robot_one = quadruped_class(t=t,q=q_d_com.reshape((1,18)),qdot=qdot_d_com.reshape((1,18)),p=p,u=tau,dt=dt,\
                            urdf_file='/home/lenovo/catkin_ws/src/legs/urdf/legs_rbdl.urdf',param=None,terrain=None)

#### constraints:
FR_foot_pos,FR_foot_vel = robot_one.computeFootState('FR',calc_velocity=True,q=q_rbdl,qdot=qdot_rbdl)
FL_foot_pos,FL_foot_vel = robot_one.computeFootState('FL',calc_velocity=True,q=q_rbdl,qdot=qdot_rbdl)
RL_foot_pos,RL_foot_vel = robot_one.computeFootState('RL',calc_velocity=True,q=q_rbdl,qdot=qdot_rbdl)
RR_foot_pos,RR_foot_vel = robot_one.computeFootState('RR',calc_velocity=True,q=q_rbdl,qdot=qdot_rbdl)
Xc_d = np.concatenate((np.concatenate((FL_foot_pos,FR_foot_pos)),np.concatenate((RL_foot_pos,RR_foot_pos))))
Xc_d = Xc_d.reshape(12,1)
Xdotc_d = np.concatenate((np.concatenate((FL_foot_vel,FR_foot_vel)),np.concatenate((RL_foot_vel,RR_foot_vel))))
Xdotc_d = Xdotc_d.reshape(12,1)


x_com_d,xdot_com_d = robot_one.get_com(q=q_d_com, qdot=qdot_d_com, body_part='robot', calc_velocity=True)
x_com_d = x_com_d.reshape(3,1)
xdot_com_d = xdot_com_d.reshape(3,1)
x_t = np.concatenate((x_com_d,Xc_d))
xdot_t = np.concatenate((xdot_com_d,Xdotc_d))

kp_x = 30*np.eye(15)
kd_x = np.eye(15)
def callback(data):
    global tpre,a,xdot_t,tpast,x_t,kd,kd,tau, Jc_cond, \
        t_vec,pose_com,orientation_com,q_d_com, qdot_d_com,\
        x_com_d,xdot_com_d,x_t,xdot_t
  
    ################################################ give ros data to rbdl ######################################
    q = data.position
    q = np.asarray(q)
    qdot = data.velocity
    qdot = np.asarray(qdot)
    q_rbdl,qdot_rbdl = ros_to_rbdl(q,qdot)
    q_com = np.hstack((pose_com, orientation_com))
    # print("COM:",q_com)
    q_rbdl = np.hstack((q_rbdl,q_com))
    # q_d_com = np.hstack((q_d, np.zeros(6)))
    qdot_rbdl = np.hstack((qdot_rbdl,np.zeros(6)))
    # qdot_d_com = np.hstack((qdot_d, np.zeros(6)))

    ####################################### calling robot's dynamic and control classes #####################################
    t = np.array([0])

    dt = .005 # step size
    
    p = [[ ]] # the contact feet
    ########################TODO: QUES: what should i do for base's q and qdot. 
    
    robot_dyn = quadruped_class(t=t,q=q_rbdl.reshape(1,18),qdot=qdot_rbdl.reshape(1,18),p=p,u=tau,dt=dt,\
                            urdf_file='/home/lenovo/catkin_ws/src/legs/urdf/legs_rbdl.urdf',param=None,terrain=None)
    # robot_con= Control(robot_dyn)
    robot_controller = Quadruped_ControlClass(robot_dyn)

    ###################################################### qddot ######################################################
    x_com,xdot_com = robot_dyn.get_com(q=q_rbdl,qdot=qdot_rbdl,body_part='robot',calc_velocity=True)
    x_com = x_com.reshape(3,1)
    # print("===========================x_com===========================")
    # print(x_com)
    xdot_com = xdot_com.reshape(3,1)
    # print("===========================xdot_com===========================")
    # print(xdot_com)
    FR_foot_pos,FR_foot_vel = robot_dyn.computeFootState('FR',calc_velocity=True,q=q_rbdl,qdot=qdot_rbdl)
    FL_foot_pos,FL_foot_vel = robot_dyn.computeFootState('FL',calc_velocity=True,q=q_rbdl,qdot=qdot_rbdl)
    RL_foot_pos,RL_foot_vel = robot_dyn.computeFootState('RL',calc_velocity=True,q=q_rbdl,qdot=qdot_rbdl)
    RR_foot_pos,RR_foot_vel = robot_dyn.computeFootState('RR',calc_velocity=True,q=q_rbdl,qdot=qdot_rbdl)
    Xc = np.concatenate((np.concatenate((FL_foot_pos,FR_foot_pos)),np.concatenate((RL_foot_pos,RR_foot_pos))))
    Xc = Xc.reshape(12,1)
    Xdotc = np.concatenate((np.concatenate((FL_foot_vel,FR_foot_vel)),np.concatenate((RL_foot_vel,RR_foot_vel))))
    Xdotc = Xdotc.reshape(12,1)
    # print("===========================xdot_c===========================")
    # print(Xdotc)
    x = np.concatenate((x_com_d,Xc_d))
    xdot = np.concatenate((xdot_com_d,Xdotc_d))
    xddot = np.dot(kp_x,(x_t-x))+np.dot(kd_x,(xdot_t-xdot))
    j_task = robot_dyn.j_task(q)
    jdotqdot = robot_dyn.calcJdQd(q_rbdl,qdot_rbdl,'task')
    qddot = np.dot(np.linalg.pinv(j_task),(xddot-jdotqdot))
    # qddot = np.concatenate((qddot,np.zeros((6,1))))
    # print("===============================qddot==============================")
    # print(qddot)
    # print(np.shape(qddot))
    # dt = time.time()- tpre
    # tpre = tpre + dt
    # xdot_t = dt*xddot + xdot_t_pre
    # xdot_t_pre = xdot_t  

    ################################################# Calctau(PID) #################################################
    error = q_d_com - q_rbdl
    error_dot = qdot_d_com - qdot_rbdl
    tau_pid = np.dot(error, kp) + np.dot(error_dot, kd)   

    ################################################# Torque calculations #################################################
    tau_qr = robot_controller.InvDyn_qr(q_des=q_d_com,qdot_des=qdot_d_com,qddot_des=qddot.flatten())
    # print(tau_pid)
    # print("============================")
    # print(tau_qr)
    # pubish(tau_qr)
    # print(robot_dyn.q[-1,:])
    # pubish(np.dot(robot_dyn.S,tau_pid))

    # robot_con.compute_torque(qdd_des=np.zeros(18),qdot_des=qdot_d,q_des=q_d)

    ############################################ Append states for plotting ############################################
    Jc_cond.append(np.linalg.cond(robot_dyn.Jc))
    t_vec.append(rospy.get_time()-tpre)

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
    rospy.Subscriber("/odom", Odometry, odom_data)
    rospy.Subscriber("/imu/data", Imu, imu_data)
    rospy.Subscriber("/legs/joint_states", JointState, callback)
    rospy.spin()
    if rospy.is_shutdown():
        plt.figure()
        plt.plot(t_vec,Jc_cond,'r')
        plt.title("jc condition number")
        plt.show()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInternalException:
        pass