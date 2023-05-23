# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:30:21 2023

@author: 12092
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

C = 6800 #kPa ## I can vary this between 5kPa to 9kPa
pw = 1000 # kg/m3
ps = 1600 # kg/m3c
phi = np.deg2rad(41)  # converts phi to radians ## set to 42 right now
g = 9.81  # m/s2
K_avg = 0.004 # m2/yr
Sc = 1.25
slope_ang = np.arange(27,51,0.1)
slope_rad = np.deg2rad(slope_ang)
hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))

Kh = []
for i in slope_rad:
    k = K_avg / (1 - ((np.tan(i)/Sc)**2))
    Kh += [k]


def crit_depth(a):
    result = (a)/(ps*g*(np.cos((hollow_rad))**2)*(np.tan((hollow_rad))-((1-(1*(pw/ps)))*np.tan((phi)))))
    return result

hc = crit_depth(6800)

ri = []
for v, i, z, k in zip(hc, Kh, slope_rad, hollow_rad):
    B = (((np.cos(k))**(1/2))*(((np.tan(z))**2)-((np.tan(k))**2))**(1/4)) * ((((np.cos(k)**2)/(np.cos(z)**2))-1)**(1/4))
    ## take out 1/2
    y = (v**2)/ ((B**2) * (2*i))
    ri += [y]
    
    

plt.figure()
plt.grid()
plt.title("Cohesion Comparison with NonLinear K")
plt.yscale('log')
plt.plot(hollow_ang, ri, label = 'C = 25.6 kPa')