import numpy as np
from numpy.linalg import multi_dot
import control
from copy import copy
import scipy.linalg as sp_linalg

class LQR_discrete(object):
    def __init__(self, A, B, Q, R, x_desired, iters, gamma=1.):
        """
            We iteratively solve K here
        """
        self.x_desired = x_desired
        # iterative solve k 
        ks_T = np.array([0., 0., 0., 0., 0., 0.])
        ks = [ks_T]         
        P_tplus1 = Q        
        for i in range(iters):
            # solve k       
            k_part1 = R + gamma*multi_dot([B.T, P_tplus1, B])
            k_part2 = multi_dot([B.T, P_tplus1, A])
            kt = gamma*multi_dot([np.linalg.inv(k_part1), k_part2])
            ks.insert(0, kt[0]) # reverse order
                            
            # update P      
            Pt_part1 = Q + multi_dot([kt.T, R, kt])
            temp = A-np.dot(B, kt)
            Pt_part2 = multi_dot([temp.T, P_tplus1, temp])
            Pt = Pt_part1 + Pt_part2
                            
            P_tplus1 = Pt   
        self.ks = copy(ks)
        self.K = ks[0]

    def apply(self, X):
        """
        [-249.00376091, -127.40804975, -2.6776243, -51.43314285, -33.83664616, -7.11248171]
        """
        K_LQR = next(self.KS)
        X = X - self.x_desired
        return -np.dot(K_LQR, X)

    def reset(self):
        self.KS = iter(self.ks)

class LQR_timevariant(object):
    def __init__(self, As, Bs, Q, R, x_desired, iters, gamma=1.):
        """
            We iteratively solve K here
        """
        self.x_desired = x_desired
        self.ks = []
        for index in range(len(As)):
            A = As[index]
            B = Bs[index]
            # iterative solve k 
            ks_T = np.array([0., 0., 0., 0., 0., 0.])
            ks = [ks_T]         
            P_tplus1 = Q        

            for i in range(iters):
                # solve k       
                k_part1 = R + gamma*multi_dot([B.T, P_tplus1, B])
                k_part2 = multi_dot([B.T, P_tplus1, A])
                kt = gamma*multi_dot([np.linalg.inv(k_part1), k_part2]) # use pseudo inverse here
                                                                         # because k_part1 is singular
                ks.insert(0, kt[0]) # reverse order
                                
                # update P      
                Pt_part1 = Q + multi_dot([kt.T, R, kt])
                temp = A-np.dot(B, kt)
                Pt_part2 = multi_dot([temp.T, P_tplus1, temp])
                Pt = Pt_part1 + Pt_part2            
                P_tplus1 = Pt   
            self.ks.append(ks[0])
        ctrb = control.ctrb(A, B)
        self.ctrb = ctrb
        self.ctrb_rank = np.linalg.matrix_rank(ctrb)

    def apply(self, X):
        K_LQR = next(self.KS)
        X = X - self.x_desired
        return -np.dot(K_LQR, X)

    def reset(self):
        self.KS = iter(self.ks)

class LQR_continuous(object):
    def __init__(self, A, B, Q, R, x_desired, iters=None):
        """
            We use some control library to solve K here
        """
        K = np.array(control.lqr(A, B, Q, R)[0])[0]
        ctrb = control.ctrb(A, B)
        self.K = K
        self.ctrb = ctrb
        self.ctrb_rank = np.linalg.matrix_rank(ctrb)
        self.x_desired = x_desired

    def apply(self, X):
        X = X - self.x_desired
        return -np.dot(self.K, X)
    
    def reset(self):
        return 

class LQR_scipy(object):
    def __init__(self, x_desired, iters=None):
        self.x_desired = x_desired

    def apply(self, A, B, Q, R, X):
        P = sp_linalg.solve_discrete_are(A, B, Q, R)
        K = np.dot(np.linalg.pinv(R + np.dot(B.T, np.dot(P, B))), np.dot(B.T, np.dot(P, A)))
        X = X - self.x_desired
        return -np.dot(K, X)[0]
    
    def reset(self):
        return
