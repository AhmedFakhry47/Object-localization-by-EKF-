'''
Some mathematical functions that are necessary during the process
'''
from math import sin,cos,atan2,sqrt
import numpy as np


def polar_to_cartesian(rho,phi,v):

    x,y = rho*cos(phi),rho*sin(phi)
    vx,vy = v*cos(phi),v*sin(phi)

    return [x,y,vx,vy]

def cartesian_to_polar(x,y,vx,vy):
    rho = sqrt(x**2+y**2)
    phi = atan2(y,x)
    v   = sqrt(vx**2+vy**2)

    return [rho,phi,v]

def RMSE(lst1,lst2):
    if len(lst1) != len(lst2):
        print("Length of two lists must be equal")
        return

    length = len(lst1)
    sum=0.0
    for i,(x,y) in enumerate(zip(lst1,lst2)):
        sum+= (x-y)**2

    return np.sqrt(sum/length)
