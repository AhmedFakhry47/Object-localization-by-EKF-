import numpy as np
'''
    Kalman filter
    X denotes vector for state variables
    F state transition model    --Fixed
    H observation transition model  --Fixed
    Q denotes process noise covarience  --Fixed
    R denotes measurement noise covarience --Fixed
    P denotes error covarience matrix ( a measure for accuracy of state estimate )
    Z denotes measurement matrix
'''
class KalmanFilter  :

    def __init__ (self,meta):
        #The four fixed matrices
        self.F = meta['F']
        self.n = self.F.size
        self.H = meta['H']
        self.m = self.H.size
        self.R = meta['R']

        #Initialization of remaining matrices
        self.X = np.zeros(shape=(self.n,1)) if ('X' not in list(meta.keys())) else meta['X']
        self.P = np.eye(self.n) if ('P' not in list(meta.keys())) else  meta['P']
        self.Q = np.eye(self.n) if ('Q' not in list(meta.keys())) else  meta['Q']
        self.Y = np.zeros(self.m).reshape(self.m,1)
        self.K = 0 #Kalman Gain

    def predict(self):
        self.X = np.dot(self.F , self.X)
        self.P = np.dot(np.dot(self.F,self.P),self.F.T) + self.Q

    def update(self,measurement,prev_polar):

        self.Y = measurement - prev_polar

        S = np.dot(self.H,np.dot(self.P,self.H.T)) + self.R
        self.K =np.dot(np.dot(self.P ,self.H.T),np.linalg.inv(S))

        #Update X and P
        self.X = self.X + np.dot(self.K,self.Y)
        self.P = self.P - np.dot(self.K,np.dot(self.H,self.P))

    def get_x(self):
        return self.X

    def eval(self,measurements):
        self.predict()
        self.update(measurements)
        return self.X
