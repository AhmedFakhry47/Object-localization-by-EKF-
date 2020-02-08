import matplotlib.pyplot as plt
from maths import RMSE
from sympy import *
from EKF import EKF
import numpy as np
import sympy



def run(sens_meas,GT):
    meta = dict()

    x,y,vx,vy,dt,t = Symbol('x'),Symbol('y'),Symbol('vx'),Symbol('vy'),Symbol('dt'),Symbol('t')

    meta['F']  = np.array([[x+vx*dt],[y+vy*dt],[vx],[vy]])
    meta['H']  = np.array([[sympy.sqrt(x**2+y**2)],[sympy.atan2(y,x)],[((x*vx+y*vy)/sympy.sqrt(x**2+y**2))]]).reshape(3,1)
    meta['Q']  = np.array([[(dt**2)/4 , 0 ,(dt**3)/2, 0],   #It's set with random value
                           [0 ,(dt**2)/4 ,0 ,(dt**3)/2 ],
                           [(dt**3)/2, 0 ,(dt**2), 0   ],
                           [0 ,(dt**3)/2 ,0 ,(dt**2)   ]])

    meta['R']  = np.array([[1,0,0],[0,0.01,0],[0,0,2]]) #Sensor measurement noise covarience
    meta['symbols'] = [x,y,vx,vy,dt,t]

    filter = EKF(meta)
    estimated_states = []

    for meas in sens_meas:
        estimated_states.append(filter.eval(meas))

    with open("sample_data",'w')as f :
        for i,(est,meas,gt) in enumerate(zip(estimated_states,sens_meas,GT)):
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(est[0][0],est[1][0],est[2][0],est[3][0],meas[0],meas[1],meas[2],meas[3],gt[0],gt[1],gt[2],gt[3]))

    print(estimated_states[-1])
    return np.array(estimated_states)


def file_reader(filename):
    for row in open(filename,'r'):
        yield row

def parse_data(filename):
    #Get the radar sensor measurements
    measurements =[]
    GT = []
    lines_gen = file_reader(filename)

    for line in lines_gen:
        line = line.split()
        if (line[0]=='R'):
            measurements.append([float(num) for num in line[1:5]])
            GT.append([float(num) for num in line[5:]])

    measurements = np.array(measurements)
    GT = np.array(GT)

    return measurements,GT

if __name__ == '__main__':

    data_file = 'data.txt'   #data dir
    sens_meas,GT = parse_data(data_file)
    estimated_states = run(sens_meas,GT)

    print('position X RMSE  : ',RMSE(GT[:,0],estimated_states[:,0]))
    print('position Y RMSE  : ',RMSE(GT[:,1],estimated_states[:,1]))
    print('position VX RMSE : ',RMSE(GT[:,2],estimated_states[:,2]))
    print('position VY RMSE : ',RMSE(GT[:,3],estimated_states[:,3]))

    print('position',RMSE(GT[:,1],sens_meas[:,1]))

    labels = ['Estimated position X','Estimated position Y','Estimated vx','Estimated Vy']
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(sens_meas[:,-1],GT[:,i],'.-r',linestyle='dashed',label='Ground Truth')
        plt.plot(sens_meas[:,-1],estimated_states[:,i],'.-b',label=labels[i])
        plt.plot(sens_meas[:,-1],sens_meas[:,i],'.-y',label='sensor measurement')
        plt.legend(loc='upper left')
    plt.show()
