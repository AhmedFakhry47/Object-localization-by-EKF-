'''
Extended Kalman Filter class that is inherited from kalman filter class
'''

from kalmanfilter import KalmanFilter
from maths import polar_to_cartesian,cartesian_to_polar
from sympy import *
import sympy
import numpy as np


class EKF (KalmanFilter):
    def __init__(self,meta):
        super().__init__(meta)
        self.sym = meta['symbols']
        self.jacobian_F  = self.__get_jacobian(meta['F'])
        self.jacobian_H  = self.__get_jacobian(meta['H'])
        self.express_Q   = meta['Q']
        self.timestamp   = 0.0
        self.initialized = False

    def __get_jacobian(self,vec_fun):
        jacob_ = np.zeros(shape=(vec_fun.size,len(self.sym[0:4])),dtype=object)
        for row,eqn in enumerate(vec_fun):
            jacob_[row] = [eqn[0].diff(i) for i in self.sym[:4]]
        return jacob_

    def __update_matrix(self,matrix,values):
        updated_ = np.zeros(shape=matrix.shape,dtype =object)
        for row,exprs in enumerate(matrix):
            for elem,expr in enumerate(exprs):
                updated_[row][elem]= expr.subs((sym,num) for sym,num in zip(self.sym,values)) if(type(expr) != int) else expr
                
        return updated_.astype(float)

    def __handle_prevstate(self,prev_state,dt,t):
        #sens_meas in cartesian
        values = [list(elem)[0] for elem in prev_state]
        values.extend([dt,t])
        return values

    def trace(self,sens_meas):
        #Notice that , Updating F , H has nothing to do with sensor measurements

        #determining time difference
        t  = sens_meas[-1]/1e6
        dt = (sens_meas[-1]-self.timestamp)/1e6
        self.timestamp = sens_meas[-1]

        #Calculate jacobian of F at new point
        self.Q = self.__update_matrix(self.express_Q,self.__handle_prevstate(self.X,dt,t))
        self.F = self.__update_matrix(self.jacobian_F,self.__handle_prevstate(self.X,dt,t))
        self.predict()

        prev_polar = np.array(cartesian_to_polar(self.X[0],self.X[1],self.X[2],self.X[3])).reshape(3,1)

        #Calculate Jacobian of H at new point
        self.H = self.__update_matrix(self.jacobian_H,self.__handle_prevstate(self.X,dt,t)) #states_polar.append(sens_meas[3]))
        self.update(np.array(sens_meas[:3]).reshape(3,1),prev_polar)

    def start(self,sens_meas):
        self.timestamp = sens_meas[-1]
        self.X = np.array(polar_to_cartesian(sens_meas[0],sens_meas[1],sens_meas[2])).reshape((self.n,1))
        self.initialized = True

    def eval(self,sens_meas):
        if self.initialized:
            self.trace(sens_meas)
        else:
            self.start(sens_meas)

        return self.X
