
from numpy import *
import numpy as np
from autograd import jacobian, numpy as anp

G =  9.8 # acceleration due to gravity, in m/s^2
R = 1.0  # length of the pole (m)
M = 4.0  # mass of the cart (kg)
m = 1.0  # mass of the ball at the end of the pole (kg)

def nonlinear_dynamics(q, u):
    delta = m*anp.sin(q[0])**2 + M
    
    f1 = q[2]
    f2 = q[3]
    
    f3 = - m*(q[2]**2)*anp.sin(q[0])*anp.cos(q[0])/delta  \
    		  - (m+M)*G*anp.sin(q[0])/delta/R  \
    		  - u*anp.cos(q[0])/delta/R
    
    f4 = m*R*(q[2]**2)*anp.sin(q[0])/delta   \
    		 + m*R*G*anp.sin(q[0])*anp.cos(q[0])/delta/R  \
    		 + u/delta
    
    return anp.array([f1, f2, f3, f4])

def linearization(fixed_point_x, fixed_point_u):
    jacobian_x = jacobian(nonlinear_dynamics, 0)
    jacobian_u = jacobian(nonlinear_dynamics, 1)
    A = jacobian_x(fixed_point_x, fixed_point_u)
    B = jacobian_u(fixed_point_x, fixed_point_u)
    return A, B

def euler_discritization(A, B, delta_t):
    Identity = np.identity(A.shape[0])
    A_prime = A*delta_t + Identity
    B_prime = B*delta_t     
    return A_prime, B_prime
            
            
            