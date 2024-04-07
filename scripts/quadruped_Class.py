# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:31:20 2022
@author: Nooshin Kohli
"""
import sys
from os.path import expanduser

home = expanduser("~")
dir = home + '/rbdl/build/python'
sys.path.append(dir)

import numpy as np
import rbdl
#from Centauro_ImportModel import BodyClass, JointClass
#import matplotlib.pyplot as plt
#from scipy.optimize import root, approx_fprime, minimize, fminbound
import scipy.integrate as integrate
#import time
#import subprocess


class quadruped_class(object):
    def __init__(self, t, q, qdot, p, u, dt, urdf_file, param=None, terrain=None):
        """
        This is quadruped robot class
        """        
        self.name = 'quadruped'
        # TODO: convert fixed to floating in rbdl model
        
        self.model = rbdl.loadModel(urdf_file)
#        self.body = BodyClass()
#        self.joint = JointClass()
        
        self.qdim = self.model.q_size
        self.S = np.hstack((np.eye(self.qdim - 6),np.zeros((self.qdim - 6, 6))))
        self.t = np.array([t])
        self.calf_length = -0.23039
        self.leg_end = np.asarray([0.0, 0.0, self.calf_length])
        self.mass_hip = 0.57
        self.mass_thigh = 0.634
        self.mass_calf = 0.064
        
        if q.any(): 
            self.q = np.array(q)
            # print(self.q)
        else: self.q = np.zeros((1, self.qdim)) 
        if qdot.any(): self.qdot = np.array(qdot) # states
        else: self.qdot = np.zeros((1, self.qdim)) 
        self.__p = list(p) # contact feet
        if u.any(): self.u = np.array(u) # joint inputs
        else: self.u = np.zeros((1, self.qdim - 6))
        self.cforce = []
        
        self.dt = dt 
        
        self.terrain = terrain
        self.calf_len = 0.23039
        
        self.M = self.CalcM(self.model, self.q[-1, :])
        self.h = self.Calch(self.model, self.q[-1, :], self.qdot[-1, :])
        self.Jc = self.Jc_from_cpoints( \
            self.model, self.q[-1, :], [1,2,3,4])
        # TODO: calculate total mass here 
#        self.total_mass = sum([self.model.mBodies[i].mMass for i in range(42)])

        
    def computeFootState(self, body_part, \
                         calc_velocity=False, update_kinematics=True, \
                         index=-1, q=None, qdot=None):
        
        point = np.array([0., 0., self.calf_length])
        if body_part == 'FR': body_id = self.model.GetBodyId('FR_calf')
        elif body_part == 'FL': body_id = self.model.GetBodyId('FL_calf')
        elif body_part == 'RR': body_id = self.model.GetBodyId('RR_calf')
        elif body_part == 'RL': body_id = self.model.GetBodyId('RL_calf')
        else:
            raise ValueError("This part does not exist!!!!")

        return self.CalcBodyToBase(body_id, point, \
                                   calc_velocity=calc_velocity, \
                                   update_kinematics=update_kinematics, \
                                   index=index, q=q, qdot=q)

    def ComputeContactForce(self, qqdot, p, u):
        
        q = qqdot[:self.qdim] 
        qdot = qqdot[self.qdim:] 
        
        
        Jc = self.Jc_from_cpoints(self.model, q, p)  
        
        M = self.CalcM(self.model, q)
        
        h = self.Calch(self.model, q, qdot)
        
        self.ForwardDynamics(qqdot.flatten(), M, h, self.S, u, Jc, p)
        
        return None
                
        

    def __call__(self):
        """
        executes hybrid system
        """
        self.t0 = self.t[-1]
        self.qqdot0 = np.concatenate((self.q[-1,:], self.qdot[-1, :])).\
        reshape(1, self.qdim*2)
        self.qqdot0forRefine = self.qqdot0[-1, :].copy()
        self.__p0 = self.__p[-1]
        
        if not hasattr(self, 'u0'): self.u0 = np.zeros_like(self.u[-1, :])
        
#        self.ComputeContactForce(self.qqdot0forRefine, self.__p0, self.u0)
#        self.cforce.append(self.Lambda) 

#        self.qqdot0 = integrate.odeint(self.__dyn, self.qqdot0[-1,:], \
#        np.array([0,self.dt]))
        
        dy = self.RK4(self.dyn_RK4)
        self.qqdot0 += dy(self.t0, self.qqdot0[-1, :], self.dt).reshape(1, self.qdim*2)
                
        self.ev_i = None
        self.evt = list(self.__evts())      
        self.ev = np.array([self.evt[i](self.t0 + self.dt, \
        self.qqdot0[-1,:]) for i in range(len(self.evt))])
        
        
        if self.ev[-1] == 0: raise ValueError('Simulation was terminated because:\
        one of the conditions in StopSimulation() is meet.')
        
        indexes = [i for i, ev in enumerate(self.ev) if ev is not None and ev>0]

        if indexes:
            print("##########################")
            print("index of the occurred events: ", indexes)
            print("at the nominal time: ", self.t0)
            print("\n")
#            print (indexes)
#            print (self.ev)
            tr_list, qqdot0_list, index_list = [], [], []
            for i in indexes:
                if i in [8, 9, 10, 11]:
                    tr , qqdot0 = self.interpl(self.evt[i])
                else:
                    tr , qqdot0 = self.refine(self.evt[i])
                tr_list.append(tr); qqdot0_list.append(qqdot0); index_list.append(i)
            
            index = np.argmin(tr_list)
            print("the one that is applied: ", indexes[index])
            print("at the real time: ", tr_list[index])
            print("##########################\n\n")
            
            self.trefined , self.qqdot0 = tr_list[index], qqdot0_list[index]
            self.ev_i = index_list[index] 
            
            self.qqdot0, self.__p0 = self.__trans()
        

        self.__AppendState()

        return None
        
    def __AppendState(self):
        self.q = np.append(self.q, [self.qqdot0[-1,:self.qdim]], axis=0)
        self.qdot = np.append(self.qdot, [self.qqdot0[-1, self.qdim:]], axis=0)
        self.__p.append(self.__p0)
        self.u = np.append(self.u, [self.u0.flatten()], axis=0)
        if self.ev_i is not None:
            self.t = np.append(self.t, [self.trefined], axis=0)
        else:
            self.t = np.append(self.t, [self.t0 + self.dt], axis=0)
            
        self.cforce.append(self.Lambda)
        return None
        
    def RK4(self, f):
        return lambda t, y, dt: (
                lambda dy1: (
                lambda dy2: (
                lambda dy3: (
                lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
                )( dt * f( t + dt  , y + dy3   ) )
    	    )( dt * f( t + dt/2, y + dy2/2 ) )
    	    )( dt * f( t + dt/2, y + dy1/2 ) )
    	    )( dt * f( t       , y         ) )
        
    
        
    def __dyn(self, x, t):
        """
        .dyn  evaluates system dynamics
        """
        q = x[:self.qdim]
        qd = x[self.qdim:]
                
        self.M = self.CalcM(self.model, q)
        self.Jc = self.Jc_from_cpoints(self.model, q, self.__p0)
        self.h = self.Calch(self.model, q, qd)
        
        self.ForwardDynamics(x, self.M, self.h, self.S, self.u0, self.Jc, self.__p0) 
        
        dx = np.concatenate((qd, self.qddot.flatten()))

        return dx
        
    def dyn_RK4(self, t, x):
        """
        .dyn  evaluates system dynamics
        """
        return self.__dyn(x, t)

    def ForwardDynamics(self, x, M, h, S, tau, Jc, cpoints):
        fdim = np.shape(Jc)[0]
        qdim = self.qdim
        q = x[:qdim]
        qdot = x[qdim:]
        
        if fdim == 0:
        
            self.qddot = np.dot(np.linalg.inv(M), np.dot(S.T, self.u0) - h).flatten()        
            self.Lambda = np.zeros(12)*np.nan
        else:
            
            
            if np.nonzero(qdot)[0].any() and Jc.any():
#                tic = time.time()
#                gamma = self.CalcGamma(cpoints, q, qdot)
                gamma = self.CalcGamma(cpoints, q, qdot)
#                print gamma - mygamma
#                toc = time.time() - tic
#                print toc
            else:
                gamma = - np.dot(np.zeros_like(Jc), qdot)
                    
            
            aux1 = np.hstack((M, -Jc.T))
            aux2 = np.hstack((Jc, np.zeros((fdim, fdim))))
            A = np.vstack((aux1, aux2))
            
            B = np.vstack(((np.dot(S.T, tau) - h).reshape(qdim, 1), \
            gamma.reshape(fdim, 1)))
            
            res = np.dot(np.linalg.inv(A), B).flatten()
            
            self.qddot = res[:-fdim]
            
            self.SetGRF(cpoints,  res[-fdim:])
            
            
#            print "======================================="            
#            print 'p, lambda:', self.__p[-1], self.Lambda
#            print "======================================="
        
        return None
        
    def SetGRF(self, p, values):
#        print 'yes', p
        last = 0
        if 1 in p:
            p_1 = last
            last += 3
        if 2 in p:
            p_2 = last
            last += 3
        if 3 in p:
            p_3 = last
            last += 3
        if 4 in p:
            p_4 = last
            last += 3
        self.Lambda = np.zeros(12)*np.nan
        if 1 in p:
            self.Lambda[:3] = values[p_1:p_1+3]
        if 2 in p:
            self.Lambda[3:6] = values[p_2:p_2+3]
        if 3 in p:
            self.Lambda[6:9] = values[p_3:p_3+3]
        if 4 in p:
            self.Lambda[9:] = values[p_4:p_4+3]
        return None
        
        
        
        
    def CalcM(self, model, q):
        M = np.zeros ((model.q_size, model.q_size))
        rbdl.CompositeRigidBodyAlgorithm(model, q, M, True)
        return M
        
    def Calch(self, model, q, qdot):
        h = np.zeros(model.q_size)
        rbdl.InverseDynamics(model, q, qdot, np.zeros(model.qdot_size), h)
        return h
        
    def CalcJacobian(self, model, q, bodyid, point):
    
        Jc = np.zeros((3, model.dof_count))
        rbdl.CalcPointJacobian (model, q, bodyid, point, Jc)
        
        return Jc
        
        
    def Jc_from_cpoints(self, model, q, cpoints):
        
        Jc = np.array([])
        
        ftip_pose = np.array([0., 0., -self.calf_len])
        
        if 1 in cpoints:
#                tic = time.time()
            Jc_ = self.CalcJacobian(model, q, self.model.GetBodyId('FL_calf'), ftip_pose)
            
            Jc = np.append(Jc, Jc_)
#                toc = time.time() - tic
#                print toc   
        if 2 in cpoints:
            Jc_ = self.CalcJacobian(model, q, self.model.GetBodyId('FR_calf'), ftip_pose)
            Jc = np.append(Jc, Jc_)
            
        if 3 in cpoints:
            Jc_ = self.CalcJacobian(model, q, self.model.GetBodyId('RL_calf'), ftip_pose)
            Jc = np.append(Jc, Jc_)
            
        if 4 in cpoints:
            Jc_ = self.CalcJacobian(model, q, self.model.GetBodyId('RR_calf'), ftip_pose)
            Jc = np.append(Jc, Jc_)
            
        return Jc.reshape(np.size(Jc)//model.dof_count, model.dof_count)


    def CalcGamma(self, cp, q, qdot):
        
        self.cbody_id = []
        
        if 1 in cp:
            for i in range(3):self.cbody_id.append(self.model.GetBodyId('FL_calf'))
        if 2 in cp: 
            for i in range(3):self.cbody_id.append(self.model.GetBodyId('FR_calf'))
        if 3 in cp: 
            for i in range(3):self.cbody_id.append(self.model.GetBodyId('RL_calf'))
        if 4 in cp: 
            for i in range(3):self.cbody_id.append(self.model.GetBodyId('RR_calf'))
        
        Normal = []
        for i in range(len(cp)):
            Normal.append(np.array([1., 0., 0.]))
            Normal.append(np.array([0., 1., 0.]))
            Normal.append(np.array([0., 0., 1.]))
        
        
        k = len(cp)*3
        
        Gamma = np.zeros(k)
        
        prev_body_id = 0
        
        gamma_i = np.zeros(3)
        
        for i in range(k):
            
            if prev_body_id != self.cbody_id[i]:
                gamma_i = rbdl.CalcPointAcceleration(self.model, q,\
                qdot, np.zeros(self.qdim), self.cbody_id[i], \
                np.array([0., 0., - self.calf_len]))                
                prev_body_id = self.cbody_id[i]
                
            Gamma[i] = - np.dot(Normal[i], gamma_i)
        return Gamma
        
        
    

    def CalcAcceleration(self, q, qdot, qddot, body_id, body_point):
        body_accel = rbdl.CalcPointAcceleration(self.model, q, qdot, qddot, \
        body_id, body_point)
        return body_accel
        
    def TerrainHeight(self, x, y):
        if self.terrain is None:
            return 0

    
    def set_input(self, tau):
        if len(tau) == self.qdim: self.u0 = np.dot(self.S, tau)
        else: self.u0 = tau
#        print self.u0
        return None
        
    def get_com(self, calc_velocity = False, calc_angular_momentum = False, \
    update = True, index = -1, body_part = 'robot', q = None, qdot = None):      
        '''
        computes position and velocity of COM of a body or whole robot
        '''
        mass = 0
        qddot = np.zeros(12)
        com = np.zeros(3)
        if calc_velocity: com_vel = np.zeros(3)
        else: com_vel = None
        if calc_angular_momentum: 
            angular_momentum = np.zeros(3)
            # print("ok")
        else: 
            angular_momentum = None
        if q is not None: qq = q
        else: qq = self.q[index, :]
        if qdot is not None: qqdot = qdot
        else: qqdot = self.qdot[index, :]
        if body_part == 'robot':
            rbdl.CalcCenterOfMass(self.model, q = qq, qdot = qqdot,\
                 com = com, qddot=qddot , com_velocity = com_vel, angular_momentum=angular_momentum, update_kinematics=update)            
        
            if calc_velocity and calc_angular_momentum:
                return com, com_vel, angular_momentum
            elif calc_velocity and not calc_angular_momentum:
                return com, com_vel
            else: return com
        else:
            com, vel = self.__calculateBodyCOM(qq, \
            qqdot, calc_velocity, update, body_part)
            if calc_velocity:
                return com, vel
            else:
                return com 
        
    
    def __calculateBodyCOM(self, q, dq, calc_velocity, update, body_part):
        if body_part == 'FR':
            p1 = self.CalcBodyToBase(self.model.GetBodyId('FR_hip'), 
                                     np.array([0.0, 0.036, 0.0]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            p2 = self.CalcBodyToBase(self.model.GetBodyId('FR_thigh'), 
                                     np.array([0.0, 0.016 ,-0.11]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            p3 = self.CalcBodyToBase(self.model.GetBodyId('FR_calf'), 
                                     np.array([0., 0., (1/2)*self.calf_length]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            
            if not calc_velocity:
                com = (self.mass_hip*p1 + self.mass_thigh*p2 + self.mass_calf*p3)/\
                  (self.mass_hip + self.mass_thigh + self.mass_calf)
                vel = None
            else:
                com = (self.mass_hip*p1[0] + self.mass_thigh*p2[0] + self.mass_calf*p3[0])/\
                      (self.mass_hip + self.mass_thigh + self.mass_calf)
                vel = (self.mass_hip*p1[1] + self.mass_thigh*p2[1] + self.mass_calf*p3[1])/\
                      (self.mass_hip + self.mass_thigh + self.mass_calf)
                
                  
        if body_part == 'FL':
            p1 = self.CalcBodyToBase(self.model.GetBodyId('FL_hip'), 
                                     np.array([0.0, 0.036, 0.0]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            p2 = self.CalcBodyToBase(self.model.GetBodyId('FL_thigh'), 
                                     np.array([0.0, 0.016 ,-0.11]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            p3 = self.CalcBodyToBase(self.model.GetBodyId('FL_calf'), 
                                     np.array([0., 0., (1/2)*self.calf_length]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            
            if not calc_velocity:
                com = (self.mass_hip*p1 + self.mass_thigh*p2 + self.mass_calf*p3)/\
                  (self.mass_hip + self.mass_thigh + self.mass_calf)
                vel = None
            else:
                com = (self.mass_hip*p1[0] + self.mass_thigh*p2[0] + self.mass_calf*p3[0])/\
                      (self.mass_hip + self.mass_thigh + self.mass_calf)
                vel = (self.mass_hip*p1[1] + self.mass_thigh*p2[1] + self.mass_calf*p3[1])/\
                      (self.mass_hip + self.mass_thigh + self.mass_calf)
        
        if body_part == 'RR':
            p1 = self.CalcBodyToBase(self.model.GetBodyId('RR_hip'), 
                                     np.array([0.0, 0.036, 0.0]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            p2 = self.CalcBodyToBase(self.model.GetBodyId('RR_thigh'), 
                                     np.array([0.0, 0.016 ,-0.11]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            p3 = self.CalcBodyToBase(self.model.GetBodyId('RR_calf'), 
                                     np.array([0., 0., (1/2)*self.calf_length]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            
            if not calc_velocity:
                com = (self.mass_hip*p1 + self.mass_thigh*p2 + self.mass_calf*p3)/\
                  (self.mass_hip + self.mass_thigh + self.mass_calf)
                vel = None
            else:
                com = (self.mass_hip*p1[0] + self.mass_thigh*p2[0] + self.mass_calf*p3[0])/\
                      (self.mass_hip + self.mass_thigh + self.mass_calf)
                vel = (self.mass_hip*p1[1] + self.mass_thigh*p2[1] + self.mass_calf*p3[1])/\
                      (self.mass_hip + self.mass_thigh + self.mass_calf)
        
        if body_part == 'RL':
            p1 = self.CalcBodyToBase(self.model.GetBodyId('RL_hip'), 
                                     np.array([0.0, 0.036, 0.0]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            p2 = self.CalcBodyToBase(self.model.GetBodyId('RL_thigh'), 
                                     np.array([0.0, 0.016 ,-0.11]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            p3 = self.CalcBodyToBase(self.model.GetBodyId('RL_calf'), 
                                     np.array([0., 0., (1/2)*self.calf_length]),
                                     update_kinematics = update,
                                     q = q, qdot = dq, calc_velocity = calc_velocity)
            
            if not calc_velocity:
                com = (self.mass_hip*p1 + self.mass_thigh*p2 + self.mass_calf*p3)/\
                  (self.mass_hip + self.mass_thigh + self.mass_calf)
                vel = None
            else:
                com = (self.mass_hip*p1[0] + self.mass_thigh*p2[0] + self.mass_calf*p3[0])/\
                      (self.mass_hip + self.mass_thigh + self.mass_calf)
                vel = (self.mass_hip*p1[1] + self.mass_thigh*p2[1] + self.mass_calf*p3[1])/\
                      (self.mass_hip + self.mass_thigh + self.mass_calf)
                
                
        return com, vel

    def computeJacobianCOM(self,q, body_part):
        '''
        This function calculates jacobian of center of mass
        center of mass is from urdf of legs
        TODO: why if??
        '''
        bis = []
        pts = []
        ms = []
        if body_part == 'FL':
            bis.append(self.model.GetBodyId('FL_hip'))
            bis.append(self.model.GetBodyId('FL_thigh'))
            bis.append(self.model.GetBodyId('FL_calf'))
            pts.append(np.array([0.0, 0.036, 0.0]))
            pts.append(np.array([0.0, 0.016 ,-0.11]))
            pts.append(np.array([0., 0., (1/2)*self.calf_length]))
            ms = [self.mass_hip, self.mass_thigh, self.mass_calf]
            
        elif body_part == 'FR':
            bis.append(self.model.GetBodyId('FR_hip'))
            bis.append(self.model.GetBodyId('FR_thigh'))
            bis.append(self.model.GetBodyId('FR_calf'))
            pts.append(np.array([0.0, 0.036, 0.0]))
            pts.append(np.array([0.0, 0.016 ,-0.11]))
            pts.append(np.array([0., 0., (1/2)*self.calf_length]))
            ms = [self.mass_hip, self.mass_thigh, self.mass_calf]

        elif body_part == 'RL':
            bis.append(self.model.GetBodyId('RL_hip'))
            bis.append(self.model.GetBodyId('RL_thigh'))
            bis.append(self.model.GetBodyId('RL_calf'))
            pts.append(np.array([0.0, 0.036, 0.0]))
            pts.append(np.array([0.0, 0.016 ,-0.11]))
            pts.append(np.array([0., 0., (1/2)*self.calf_length]))
            ms = [self.mass_hip, self.mass_thigh, self.mass_calf]

        elif body_part == 'RR':
            bis.append(self.model.GetBodyId('RR_hip'))
            bis.append(self.model.GetBodyId('RR_thigh'))
            bis.append(self.model.GetBodyId('RR_calf'))
            pts.append(np.array([0.0, 0.036, 0.0]))
            pts.append(np.array([0.0, 0.016 ,-0.11]))
            pts.append(np.array([0., 0., (1/2)*self.calf_length]))
            ms = [self.mass_hip, self.mass_thigh, self.mass_calf]
        
            
        body = ['FL','FR','RL','RR']

        J = np.zeros((3, self.qdim))
        for j,name in enumerate(body):
            for i in range(len(ms)):
                if i==0: bi= self.model.GetBodyId(name+"_hip")
                elif i==1: bi = self.model.GetBodyId(name+"_thigh")
                elif i ==2: bi = self.model.GetBodyId(name+"_calf")
                J += ms[i]*self.CalcJacobian(self.model, q, bi, pts[i])

        
        # for i, bi in enumerate(bis):
        #     J += ms[i]*self.CalcJacobian(self.model, self.q, bi, pts[i])
            
        return J[:3, :]/sum(ms)
    

    def j_task(self, q):
        '''
        task to joint space jacobian
        '''
        j_com = self.computeJacobianCOM(q,"RR")
        jc = self.Jc_from_cpoints(self.model, q, [1,2,3,4])
        j_task = np.vstack((j_com, jc))
        return j_task   
        
        
    def CalcBodyToBase(self, body_id, body_point_position, \
    calc_velocity = False, update_kinematics=True, index = -1, q = None, qdot = None):
        if q is not None: qq = q
        else: qq = self.q[index, :]
        pose = rbdl.CalcBodyToBaseCoordinates(self.model, qq, \
            body_id, body_point_position, update_kinematics)
        if not calc_velocity: return pose
        else:
            if qdot is not None: qqdot = qdot
            else: qqdot = self.qdot[index, :]
            vel = rbdl.CalcPointVelocity(self.model, qq, \
            qqdot, body_id, body_point_position, update_kinematics)
            return pose, vel

    def GetContactFeet(self, total = False):
        if not total: return self.__p[-1]
        else : return self.__p
    

    def calcJdQd(self,q,qdot,body,point=None):
        '''
        TODO: if body is something other than task, desired point must be given 
        '''
        if body=='task':
            p = self.get_com('robot',q=q,qdot=qdot)
            p= np. asarray(p)
            # print(p)
            p_calf = rbdl.CalcBaseToBodyCoordinates(self.model,q,self.model.GetBodyId('FR_calf'),p[0])
            acc_com = rbdl.CalcPointAcceleration(self.model,q,qdot,np.zeros(self.qdim),self.model.GetBodyId('FR_calf'),p_calf)
            acc_com = acc_com.reshape(3,1)
            ###########
            FL_acc =  rbdl.CalcPointAcceleration(self.model,q,qdot,np.zeros(self.qdim),self.model.GetBodyId('FL_calf'),self.leg_end)
            FR_acc =  rbdl.CalcPointAcceleration(self.model,q,qdot,np.zeros(self.qdim),self.model.GetBodyId('FR_calf'),self.leg_end)
            RL_acc =  rbdl.CalcPointAcceleration(self.model,q,qdot,np.zeros(self.qdim),self.model.GetBodyId('RL_calf'),self.leg_end)
            RR_acc =  rbdl.CalcPointAcceleration(self.model,q,qdot,np.zeros(self.qdim),self.model.GetBodyId('RR_calf'),self.leg_end)
            acc_c = np.concatenate((np.concatenate((FL_acc,FR_acc)),np.concatenate((RL_acc,RR_acc))))
            acc_c =acc_c.reshape(12,1)
            body_acc = np.concatenate((acc_com,acc_c))
        else:
            id = self.model.GetBodyId(body)
            body_acc = rbdl.CalcPointAcceleration(self.model,q,qdot,np.zeros(self.qdim),id,point)
        
        return body_acc 
        
#    def SetContactFeet(self, newp):
#        self.__p[-1] = newp
#        return None    
    

   
#    def __evts(self): return [lambda t,x : -1]   
    
    def __evts(self):
        """
        x = qqdot0
        """
        p = self.__p[-1]
        return [
            lambda t, x: None if 1 in p else self.Touchdown(t, x, 1), 
            lambda t, x: None if 2 in p else self.Touchdown(t, x, 2),
            lambda t, x: None if 3 in p else self.Touchdown(t, x, 3),
            lambda t, x: None if 4 in p else self.Touchdown(t, x, 4),
            lambda t, x: None if 1 not in p else self.Liftoff(t, x, 1),
            lambda t, x: None if 2 not in p else self.Liftoff(t, x, 2),
            lambda t, x: None if 3 not in p else self.Liftoff(t, x, 3),
            lambda t, x: None if 4 not in p else self.Liftoff(t, x, 4),
            lambda t, x: None if 1 not in p else self.Liftoff_GRF(t, x, 1),
            lambda t, x: None if 2 not in p else self.Liftoff_GRF(t, x, 2),
            lambda t, x: None if 3 not in p else self.Liftoff_GRF(t, x, 3),
            lambda t, x: None if 4 not in p else self.Liftoff_GRF(t, x, 4),
            lambda t, x: None if not self.StopSimulation(t, x, p) else 0]
            
            
    def StopSimulation(self, t, x, p):
        out = False # change it if you want to stop simulation for some reasons  
        
        if len(p) < 3: 
#            out = True
            print ('less than 3 legs are in contact with the ground!')
            
        return out
#    
    def Touchdown(self, t, x, leg):
        """
        should return a positive value if leg penetrated the ground
        """
        q = x[:self.qdim]
        point = np.array([0., 0., -self.calf_len])
        if leg== 1: name = "FL"
        if leg == 2 : name = "FR"
        if leg ==3 : name = "RL"
        if leg == 4 : name = "RR" 
            
        exec("body_id = self.model.GetBodyId('"+name+"_calf')")
            
        pose = self.CalcBodyToBase(body_id, point, q = q)
        return - (pose[2] - self.TerrainHeight(pose[0], pose[1]))
        
    def Liftoff(self, t, x, leg):
        return -1
        
#    def Liftoff_GRF(self, t, x, leg, p):
#        index = p.index(leg)
##        print 'index in liftoff', index
##        print abs(t - self.t[-1]) <= self.dt
##        abs(t - self.t[-1]) <= self.dt*1.05:
#        if np.allclose(t, self.t[-1]): return - self.cforce[-1][index*3 + 2]
##        else:  return  - self.Lambda[index*3 \
##        + 2]
#        
#        elif t > self.t[-1]: return - self.Lambda[index*3 + 2]
#        else: raise ValueError('Liftoff_GRF() does not have relevant sol.')
##        else: raise ValueError('Liftoff_GRF() does not have relevant sol.') 
        
        
#    def Liftoff_GRF(self, t, x, leg, p):
#        return -1 
        
    def Liftoff_GRF(self, t, y, leg):
        if hasattr(self, 'for_refine'): u = self.u[-1, :]
        else:
            yprev = np.concatenate((self.q[-1, :], self.qdot[-1, :]))
            if np.allclose(y, yprev): u = self.u[-1, :]
            else: u = self.u0 
#        index = self.__p0.index(leg)
        self.ComputeContactForce(y, self.__p0, u)
        return - self.Lambda[(leg - 1)*3 + 2]
        
        
        
        
    def refine( self, evtFun, tol=1e-5, *args ):
        """
           Find the time at which the trajectory is a root of the
           function evtFun.
           
           The original code is part of the package 'integro', which is adopted
           here for notation consistency.
        
           evtFun is a function taking (t,y,*args) and returning a real number.
           refine() uses trajectory patch to find the "time"
           at which evtFun(t,y(t),*args)==0.
           
           This code requires that the signs of evtFun in the beginning and
           end of the patch are opposite. It finds the root by bisection 
           -- a binary search.
           
           returns t,y --the end time and state; t is correct to within +/- tol
        """
        t0, t1 = self.t0, self.t0 + self.dt
        for k in xrange(4):
            y = self.qqdot0forRefine.copy()
            f0 = evtFun(t0, y, *args)
            dy = self.RK4(self.dyn_RK4)
            y += dy(t0, self.qqdot0forRefine, np.abs(t1 - t0)).flatten()
            f1 = evtFun(t1, y, *args)
            print ('t0,f0,t1,f1', t0,f0,t1,f1)
            if f0*f1 <= 0:
                break
            t0,t1 = t0-(t1-t0)/2,t1+(t1-t0)/2
            print ("WARNING: function did not change sign -- extrapolating order ",k)
        if f1<0:
            t0,f0,t1,f1 = t1,f1,t0,f0
            
        
        
        ypre = self.qqdot0forRefine.copy()
        
#        print t0, self.CalcBodyToBase(self.body.id('pelvis'), np.zeros(3), \
#            q = ypre[:self.qdim])[0] 
        dy = self.RK4(self.dyn_RK4)
#        time = t0 + self.dt
        time = t1 + .001
#        lambda_pre = self.cforce[-1]
        del self.Lambda
        y = ypre + dy(t0, ypre, np.abs(time - t0)).flatten()   
#        print np.allclose(self.cforce[-1], lambda_pre)
#        print 'self.Lambda', self.Lambda
        self.for_refine = True
        print ('f1_new', evtFun(time, y,  *args))
#        print t1, self.CalcBodyToBase(self.body.id('pelvis'), np.zeros(3), \
#            q = y[:self.qdim])[0] 
        while abs(t1-t0)>tol:
            t = (t1+t0)/2.
      
#            dy = self.RK4(self.dyn_RK4)
#            ytest = ypre + dy(0, ypre, self.dt*0)
#            ftest = evtFun(t0 + self.dt*.001, ytest, *args )
            
#            print 'ftest', ftest
#            print 'ttest, t', ttest, t

            dy = self.RK4(self.dyn_RK4)
            y = ypre + dy(t0, ypre, np.abs(t - t0))
#            print 'lambda', self.Lambda[-4]

#            print self.t[-1], t0
#            y = integrate.odeint(self.dyn, ypre, \
#            np.array([0, np.abs(t1 - t0)/2]))[-1, :]
#            print y.shape
            f = evtFun(t, y, *args)
            print ('t*****f', t, f)
            print (self.Lambda[8])
#            print '++t', t
#            
#            print t, self.CalcBodyToBase(self.body.id('pelvis'), np.zeros(3), \
#            q = y[:self.qdim])[0]            
            
            
            
            
            if f==0:
                break
            if f>0:
                t1,f1 = t,f
            else:
                t0,f0 = t,f
                ypre = y
        del self.for_refine
        return (t1+t0)/2, y.reshape(1, self.qdim*2)
        
    def interpl(self, evtFun, *args):       
        t0, t1 = self.t0, self.t0 + self.dt
        y0 = self.qqdot0forRefine.copy()
        f0 = evtFun(t0, y0, *args)
        dy = self.RK4(self.dyn_RK4)
        y1 = y0 + dy(t0, y0, np.abs(t1 - t0)).flatten()
        f1 = evtFun(t1, y1, *args)
        print ('t0,f0,t1,f1', t0,f0,t1,f1)
        if np.abs(f0)<np.abs(f1): self.for_refine = True
        t = t0 - f0*(t1 - t0)/(f1 - f0)
        y = y0 + dy(t0, y0, np.abs(t - t0)).flatten()
        if hasattr(self, 'for_refine'): del self.for_refine
        return t, y.reshape(1, self.qdim*2)
    


   
    def __trans(self):
        p0 = list(self.__p0)
        qqdot0 = self.qqdot0[-1, :].copy()
        q, qdot = qqdot0[:self.qdim], qqdot0[self.qdim:]
        touchdown = False
        
        for i in range(4): 
            if self.ev_i == i: p0.append(i + 1); touchdown = True
        if touchdown: qdot = self.UpdateQdotCollision(q, qdot, p0)
        
        for i in [8, 9, 10, 11]:
            if self.ev_i == i and i - 7 in p0: p0.remove(i - 7)
            
            
        p0.sort()
        qqdot0_new = np.concatenate((q, qdot)).reshape(1, self.qdim*2)             
        return qqdot0_new, p0
        
        
    def UpdateQdotCollision(self, q, qdot, p0, W = None):
        J = self.Jc_from_cpoints(self.model, q, p0)
        if W is None: W = self.CalcM(self.model, q)
        invW = np.linalg.inv(W)
        aux1 = np.dot(invW, J.T)
        aux2 = np.linalg.inv(np.dot(J, np.dot(invW, J.T)))
        invJ = np.dot(aux1, aux2)        
        qdot_after = np.dot(np.eye(np.size(qdot)) - np.dot(invJ, J), qdot)
        return qdot_after
    
    def __trans2(self):
        """
        .__trans  transition between discrete modes
        """
        q, xh, xf, leg = self.q0.copy()
        m, I, l, lg, g = self.param
        m1, m2, m3, m4, m5 = m
        I1, I2, I3, I4, I5 = I
        l1, l2, l3, l4, l5 = l
        lg1, lg2, lg3, lg4, lg5 = lg
        x = self.x0[-1, :].copy()
        z, dz = x[:5], x[5:]
        e = self.e

        # SS after vlo
        if q == 0:
#            from Matrices5D import MHc
            if any([e==i for i in range(1, len(self.ev))]): q = -1
            # Touchdown
            elif e == 0:
                self.td_proxy = 0
                dz = dz.reshape(5, 1)
                dz = np.dot(self.N(), dz)
                dz = self.DSRefine(z, dz.T[-1, :]).reshape(5, 1)
                xh = xf
                xf = MHc(np.hstack((z, dz.T[-1, :])), self.param)[0] + xh
                q = 1
           
        
        x = np.array(np.hstack((z.reshape(1, 5), dz.reshape(1, 5))))
        qq = np.array([q, xh, xf, leg])

        return x, qq
        
        
    def temp(self, i):
        
        self.dy = self.RK4(self.dyn_RK4)

        import matplotlib.pyplot as plt
    
    
        def create(q, qdot): 
            return np.concatenate((q, qdot)).reshape(1, self.qdim*2)
        
        q = self.q[i, :]
        qdot = self.qdot[i, :]
        
        y = create(q, qdot)
        
        plt.plot(self.Lambda, '--*')
        
        self.dy(self.t[i], y.flatten(), self.dt)
        
        plt.plot(self.Lambda, '-o')