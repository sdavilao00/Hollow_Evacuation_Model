# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:55:10 2023

@author: 12092
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

C = 6000 #kPa ## I can vary this between 5kPa to 9kPa
pw = 1000 # kg/m3
ps = 1800 # kg/m3c
phi = np.deg2rad(38)  # converts phi to radians ## set to 42 right now
g = 9.81  # m/s2
K_avg = 0.004 # m2/yr
Sc = 1.25
slope_ang = np.arange(30,50)
slope_rad = np.deg2rad(slope_ang)
hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
t = np.arange(0,100000, 100)

def K(a):
    b = a / (1 - ((np.tan(slope_rad)/Sc)**2))
    return b

Kh = K(K_avg)

def crit_depth(a):
    result = (a)/(ps*g*(np.cos((hollow_rad))**2)*(np.tan((hollow_rad))-((1-(0.8*(pw/ps)))*np.tan((phi)))))
    return result

hc_norm = crit_depth(6000)

def ri(a):
    B = (((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))
    A = (a**2)/ ((B**2) * (2*(Kh)))
    return A

ri_norm = ri(hc_norm)

plt.figure()
plt.grid()
plt.title("Cohesion Comparison with new K")
plt.yscale('log')
plt.plot(slope_ang, ri_norm, label = 'C = 6.0 kPa')
plt.legend()
plt.xlabel('Hollow Slope (degrees)')
plt.ylabel('Recurrance Interval (yrs)')

B = (((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))

