
#
#                O ball: m = 1 kg
#               /
#              /
#             /  pole: R = 1 m
#            /
#     ______/_____
#    |            | Cart: M = 4 kg
#    |____________|
#      O        O
#
#
from numpy import *
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def derivate(q, t, u):
    """
    q: transpose([theta, x, d(theta)/dt, dx/dt])
    u: horizontal force on the cart
    """
    dqdt = np.zeros_like(q)
    dqdt[0] = q[2]
    dqdt[1] = q[3]

    # the expanded form of the generalized acceleration
    delta = m*sin(q[0])**2 + M
    
    dqdt[2] = - m*(q[2]**2)*sin(q[0])*cos(q[0])/delta  \
    		  - (m+M)*G*sin(q[0])/delta/R  \
    		  - u*cos(q[0])/delta/R
    
    dqdt[3] = m*R*(q[2]**2)*sin(q[0])/delta   \
    		 + m*R*G*sin(q[0])*cos(q[0])/delta/R  \
    		 + u/delta
 
    return dqdt

class Cartpole(object):
    def __init__(self, delta_time, sample_freq, initial_state, reset_type=None):
        self.delta_time = delta_time
        self.sample_freq = sample_freq
        self.initial_state = initial_state
        self.reset_type = reset_type

    def reset(self):
        return self.initial_state

    def step(self, state, ut):
        """
        Integral for one step. There are three kinds of integrator:
        1. scipy integrator: very fast and accurate
        2. forward euler: deltaT should be very small to make it accurate
        3. backward euler: the worst case
        """
        t = np.array([0., self.delta_time])
        y = integrate.odeint(derivate, state, t, args=(ut,))
        next_state = y[-1]
        return next_state

    def render(self, y, save=None):
        """
        Render the input trajectories
        :param: y, np array of (Horizon, 4)
        :param: save, a string variable to select the save format
        """
        #animation generation
        ## the pos of the cart
        end_time = self.delta_time*len(y)
        t = np.arange(0.0, end_time, self.delta_time)
        x1 = y[:,1]
        y1 = 0.0

        ## the pos of the pendulum
        x2 = R*sin(y[:,0]) + x1
        y2 = -R*cos(y[:,0]) + y1

        fig = plt.figure()
        ax = fig.add_subplot(121, autoscale_on=False, aspect='equal',\
        					 xlim=(-4, 4), ylim=(-4, 4))
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            thisx = [x1[i], x2[i]]
            thisy = [y1, y2[i]]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template%(i*dt))
            return line, time_text

        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
            interval=30, blit=True, init_func=init)

        ax = fig.add_subplot(322)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.grid()
        ax.plot(t,y[:,1])

        ax = fig.add_subplot(324)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.grid()
        ax.plot(t,x2)

        ax = fig.add_subplot(326)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.grid()
        ax.plot(t,y[:,0])

        plt.subplots_adjust()
        plt.show()

        if save == "mp4":
            ani.save('cart-pole-LQR.mp4', fps=30)
        elif save == "gif":
            ani.save('cart-pole.gif', writer='PillowWriter', fps=30)

def obtain_LQR(delta_time):
    """
    Obtain an LQR controller
    :param: delta_time is the time interval for intergration
    """
    from tools import linearization, euler_discritization
    from controller import LQR_discrete, LQR_continuous
    fixed_point_x = np.array([180*rad, 0.0, 0., 0.0])
    fixed_poiot_u = 0.
    A, B = linearization(fixed_point_x, fixed_poiot_u)
    A, B = euler_discritization(A, B, delta_time)
    B = np.array([
                [B[0]],
                [B[1]],
                [B[2]],
                [B[3]]
             ])
    Q = np.array([
                    [100, 0, 0, 0],
                    [0,  100, 0, 0],
                    [0,   0, 10,0],
                    [0,  0,  0, 1]
                ])
    R = np.array([[1]])
    lqr = LQR_discrete(A, B, Q, R, fixed_point_x, 2000)
    return lqr

if __name__ == "__main__":
    rad = np.pi/180
    G =  9.8 # acceleration due to gravity, in m/s^2
    R = 1.0  # length of the pole (m)
    M = 4.0  # mass of the cart (kg)
    m = 1.0  # mass of the ball at the end of the pole (kg)

    dt = 0.03 # integral inteval 

    # initial conditions
    theta = 140*rad 
    x = 0.0
    dtheta = 0.0
    xdot = 0.0
    
    # initial state of the env
    state = np.array([theta, x, dtheta, xdot])

    # setup the LQR controller, don't forget to reset the controller
    # before apply it
    lqr = obtain_LQR(dt)
    lqr.reset()
    
    # starts the environment, control with an LQR time-invariant controller
    env = Cartpole(dt, 1, state)
    st = env.reset()
    sts = []

    for i in range(500):
        u = lqr.apply(st)         
        st = env.step(st, u)
        sts.append(st)
    sts = np.array(sts)
    env.render(sts)