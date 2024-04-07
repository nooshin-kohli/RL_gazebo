
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 22:00:23 2022
@author: nooshin Kohli
"""
import numpy as np
from numpy.linalg import inv, pinv
from scipy.linalg import sqrtm
import sys
from os.path import expanduser
home = expanduser("~")
dir = home + '/rbdl/build/python'
sys.path.append(dir)
import rbdl

class Quadruped_ControlClass(object):
    def __init__(self, robot):
        """
        This is Quadruped Controllers class
        """        
        self.name = 'Quadruped_Control'
        
        self.robot = robot  
        
        self.m = self.robot.qdim
        
        self.n = 12 #len(self.robot.getContactFeet())*2
        
        self.Sc = np.hstack((np.eye(self.n), np.zeros((self.n, self.m - self.n))))
        self.Su = np.hstack((np.zeros((self.m - self.n, self.n)), np.eye(self.m - self.n)))
        
            

    def QRDecomposition(self):
        
#        JT = self.robot.Jc.T
        JT = (self.robot.Jc_from_cpoints(\
        self.robot.model, self.robot.q[-1, :],[1,2,3,4])).T
        # we put all contact feets in first try
#TODO:        self.robot.GetContactFeet()).T
        print("jacobin.T is: ", JT)
        m, n = JT.shape
        # print("m is: ",m)
        # print("n is:",n)
        
        if m == 0 or n == 0:
            raise TypeError('Try to calculate QR decomposition, while there is no contact!')
        
        from scipy.linalg import qr
        
        self.qr_Q, qr_R = qr(JT)
        
        self.qr_R = qr_R[:n, :]
        
        return None
        
    def CalcPqr(self): 
        
        self.QRDecomposition()
        
        return np.dot(self.Su, self.qr_Q.T)
    
        
    def InverseDynamics(self, qddot_des, P, W = None, tau_0 = None):
        S = self.robot.S
        M = self.robot.CalcM(self.robot.model, self.robot.q[-1, :])
        h = self.robot.Calch(self.robot.model, self.robot.q[-1, :], \
        self.robot.qdot[-1, :])
        
        
        # if P is None: P = np.eye(M.shape[0])
        if W is None: W = np.eye(S.shape[0])
        if tau_0 is None: tau_0 = np.zeros(S.shape[0])
        
        invw = inv(W)
        Mqh = np.dot(M, qddot_des) + h
            
        if self.n >= 3:        
            aux1 = np.dot(invw, np.dot(S, P.T))
            # aux1 = np.dot(invw, P.T)
            aux21 = np.dot(P, np.dot(S.T, invw))
            # aux21 = np.dot(P, invw)
            aux22 = np.dot(S, P.T)
            # aux22 = P.T
            aux2 = np.dot(aux21, aux22)
            winv = np.dot(aux1, pinv(aux2))
        else:
            w_m_s = np.linalg.matrix_power(sqrtm(W), -1)
            aux = pinv(np.dot(P, np.dot(S.T, w_m_s)))
            winv = np.dot(w_m_s, aux)
            
        aux3 = np.dot(np.eye(S.shape[0]) - np.dot(winv, np.dot(P, S.T)), invw)
        # aux3 = np.dot(np.eye(S.shape[0]) - np.dot(winv,P), invw)
        return np.dot(winv, np.dot(P, Mqh)).flatten() + np.dot(aux3, tau_0)
            
        
        
        
        
        
    def InvDyn_qr(self, q_des, qdot_des,  qddot_des, W = None,tau0=None):
        
        n = 12
        if self.n != n:
            self.n = n
            self.Sc = np.hstack((np.eye(self.n), np.zeros((self.n, self.m - self.n))))
            self.Su = np.hstack((np.zeros((self.m - self.n, self.n)), \
            np.eye(self.m - self.n)))

        kp = 4
        kd = 1
        
        self.QRDecomposition()
        
        # if W is not None:
        #     middle = np.zeros((8,8))
        #     middle[:4,:4] = np.matmul(np.matmul(np.linalg.inv(self.qr_R.T),W),np.linalg.inv(self.qr_R))
        #     middle[4:,4:] = np.eye(4)
        #     Wc = np.matmul(np.matmul(self.qr_Q,middle),self.qr_Q.T)
        #     biddle = np.zeros((8,4))
        #     biddle[:4,:] = np.linalg.inv(self.qr_R.T)
        #     bc = np.matmul(np.matmul(self.qr_Q,biddle),tau0)
        #     W = Wc
        #     tau0 = -bc
        
#        print kd
        
        # Jc = self.robot.Jc
        # Jc = self.robot.Jc_from_cpoints(self.robot.model,self.robot.q[-1],\
        #                                 self.robot.body,self.robot.getContactFeet())
        # xdot = np.matmul(Jc,qdot_des)
        # if(np.allclose(xdot,np.zeros((4,1)),atol = 0.1)):
        #     pass
        # else:
        #     raise Warning('Jc x qdot_des is not equal to zero \n xdot : {} '\
        #                     .format(xdot))
        
        p_qr = self.CalcPqr()
        
#        print p_qr.shape
        
        invdyn = self.InverseDynamics(qddot_des, p_qr, W, tau0)
        # print(invdyn)

        pd_term = kp * (q_des - self.robot.q[-1, :]) + \
        kd * (qdot_des - self.robot.qdot[-1, :])
        # print('pd' , pd_term)
        
        return np.dot(self.robot.S.T, invdyn) 
        # kp * (q_des - self.robot.q[-1, :]) + \
        # kd * (qdot_des - self.robot.qdot[-1, :])

        
        
    # some small modification in InvDyn_qr for gravity compensation    
    def Gravity_Compensation(self,q_des,qdot_des):
        n = len(self.robot.getContactFeet())*2
        if self.n != n:
            self.n = n
            self.Sc = np.hstack((np.eye(self.n), np.zeros((self.n, self.m - self.n))))
            self.Su = np.hstack((np.zeros((self.m - self.n, self.n)), \
            np.eye(self.m - self.n)))

        
        kp = 4.
        kd = kp/10
        
#        print kd
        
        p_qr = self.CalcPqr()
        
        G = self.robot.Calch(self.robot.model,self.robot.q[-1],self.robot.qdot[-1])
        l = np.linalg.pinv(np.matmul(p_qr,self.robot.S.T))
        r = np.matmul(p_qr,G)
        torque = np.matmul(l,r)
        invdyn = torque

        
        return np.dot(self.robot.S.T, invdyn) + \
        kp * (q_des - self.robot.q[-1, :]) + \
        kd * (qdot_des - self.robot.qdot[-1, :])
        
        
        
    def Opr_Space(self, J, x_des, xdot_des, xddot_des, x, xdot, \
    tau_0 = None, tau_C = None):
        """
        implements Mistry/Righetti method.
        It does not seem efficient and elegant as it requires multiple evaluations
        of inverse of Intertia matrix
        TODO:
        1. Calculate derivatives of Jacobians: Jcdot, and Jdot
        """
        
        Jc = self.robot.Jc_from_cpoints(\
        self.robot.model, self.robot.q[-1, :],[1,2,3,4])
#        TODO: 
#        self.robot.GetContactFeet())
        
        M = self.robot.CalcM(self.robot.model, self.robot.q[-1, :])
        h = self.robot.Calch(self.robot.model, self.robot.q[-1, :], \
        self.robot.qdot[-1, :])
        
        pinv_Jc = pinv(Jc)
        I = np.eye(self.robot.qdim)
        P = I - np.dot(pinv_Jc, Jc)  
        Mc = np.dot(P, M) + I - P
        inv_Mc = inv(Mc)
        C = - np.dot(pinv_Jc, Jcdot)
        Ac = inv(np.dot(J, np.dot(inv_Mc, np.dot(P, J.T))))
        aux_j = np.dto(J, np.dot(inv_Mc, P))
        J_T_hash = np.dot(Ac, aux_j)
        N = I - np.dot(J.T, J_T_hash)
        
        kp = 4.
        kd = kp/5
        xddot = xddot_des + kd*(xdot_des - xdot) + kp*(x_des - x)
        
        aux1 = np.dot(J, np.dot(inv_Mc, np.dot(P, h)))
        aux2 = np.dot(Jdot, self.robot.qdot[-1, :]) + \
        np.dot(J, np.dot(inv_Mc, C))
        aux = aux1 - aux2
        F = np.dot(Ac, xddot) + np.dot(Ac, aux)
        
        if np.isnan(tau_0).all(): tau_0 = np.zeros(M.shape[0] - 6)
        if np.isnan(tau_C).all(): tau_C = np.zeros(M.shape[0] - 6)
        tau = np.dot(P, np.dot(J.T, F)) + np.dot(P, np.dot(N, tau_0)) + \
        np.dot(I - P, tau_C)
        
        return tau
        
        
class Control():
    
    def __init__(self,robot):
        self.name = 'Quadruped_Control'
        
        self.robot = robot  
        
        self.f = 6                                             #TODO: ques: f was 3 i put 6 
        self.n = self.robot.qdim - self.f
        
        self.k = len([1,2,3,4])*3 #2                              #TODO: this should be len(GetContactfeet())
        # print(self.k,"tihs is k")
        

        
        self.Sc = np.hstack((np.eye(self.k),np.zeros((self.k,self.n+self.f-self.k))))
        self.Su = np.hstack((np.zeros((self.n+self.f-self.k,self.k)),\
                             np.eye(self.n+self.f-self.k,self.n+self.f-self.k)))
        
        
        self.S = self.robot.S
    
    def Calc_QR(self):
        Jc = self.robot.Jc_from_cpoints(self.robot.model,self.robot.q[-1,:],[1,2,3,4])
                                        #self.robot.body,self.robot.getContactFeet())
        # print("===========================")
        # print("jc:", self.robot.q[-1,:])  
        Q,R = np.linalg.qr(Jc.T,mode="complete")
        R = R[:self.k,:]
        return Q,R
    
    def Calc_Mqh(self,qddot_des):
        M = self.robot.CalcM(self.robot.model, self.robot.q[-1, :])
        h = self.robot.Calch(self.robot.model, self.robot.q[-1, :], \
        self.robot.qdot[-1, :])
        # print(self.robot.q[-1,:])
        
        Mqh = np.dot(M, qddot_des) + h
        return Mqh
    
    def compute_torque(self,qdd_des,qdot_des,q_des):
        Mqh = self.Calc_Mqh(qdd_des)
        Q,R = self.Calc_QR()
        Q_c = Q[ :, :self.k]
        Q_u = Q[:, self.k: ]
        W = np.eye(12)
        ##############################################################
        # A=np.dot(Q_u.T,np.dot(self.S.T,np.dot(np.linalg.inv(W),np.dot(self.S,Q_u))))
        # # print(Q_u.T)
        # B = np.dot(np.linalg.inv(W),np.dot(self.S,np.dot(Q_u,np.linalg.inv(A))))
        # tau = np.dot(B,np.dot(Q_u.T,Mqh))
        # print(tau)
        ##############################################################
        fp = np.dot(np.linalg.inv(W),np.linalg.pinv(np.dot(Q_u.T,np.dot(self.S.T,np.linalg.inv(W)))))
        tau = np.dot(fp,np.dot(Q_u.T,Mqh))
        print(fp)
        ##############################################################
        ## fpp = np.matmul(self.S.T,np.matmul(self.Su,Q))
        # fp = np.matmul(np.matmul(self.Su,Q.T),self.S.T)
        # # ff = np.matmul(fp,fpp)
        # # print(Q)
        # sp = np.matmul(self.Su,Q.T)
        # tau = np.matmul(np.matmul(np.linalg.pinv(fp),sp),Mqh)
        
        kp = 10.
        kd = 10
        #print("tau shape: ",np.shape(tau))
        #print("shape: ",kp * np.matmul(self.robot.S,(q_des - self.robot.q[-1, :])))
        return tau #+ kp * np.matmul(self.robot.S,(q_des - self.robot.q[-1, :])) + \
        #kd * np.matmul(self.robot.S, (qdot_des - self.robot.qdot[-1, :]))
        