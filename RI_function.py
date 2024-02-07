# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:30:21 2023

@author: Selina Davila Olivera
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
slope_ang = np.arange(27,51, 0.1)
slope_rad = np.deg2rad(slope_ang)
hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
t = np.arange(0,100000, 100)

def K(a):
    b = a / (1 - ((np.tan(slope_rad)/Sc)**2))
    return b

Kh = K(K_avg)

def crit_depth(a):
    result = (a)/(ps*g*(np.cos((hollow_rad))**2)*(np.tan((hollow_rad))-((1-(1*(pw/ps)))*np.tan((phi)))))
    return result

hc_norm = crit_depth(6800)
hc_low = crit_depth(2500)
hc_med = crit_depth(4500)
hc_high = crit_depth(11000)
hc_max = crit_depth(25600)

def ri(a):
    B = (((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))
    A = (a**2)/ ((B**2) * (2*(Kh)))
    return A

ri_norm = ri(hc_norm)
ri_low = ri(hc_low)
ri_med = ri(hc_med)
ri_high = ri(hc_high)
ri_max = ri(hc_max)

plt.figure()
plt.grid()
plt.title("Cohesion Comparison")
plt.yscale('log')
plt.plot(hollow_ang, ri_max, label = 'C = 25.6 kPa')
plt.plot(hollow_ang, ri_high, label = 'C = 11 kPa')
plt.plot(hollow_ang, ri_norm, label = 'C = 6.8 kPa')
plt.plot(hollow_ang, ri_med, label = 'C = 4.5 kPa')
plt.plot(hollow_ang, ri_low, label = 'C = 2.5 kPa')
plt.legend()
plt.xlabel('$θ_H$')
plt.ylabel('Recurrance Interval (yrs)')

##############################################################################
######################### Linear K ###########################################
##############################################################################


hc_norm_avg = crit_depth(6800)


def ri(a):
    B = (((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))
    A = (a**2)/ ((B**2) * (2*(K_avg)))
    return A

ri_norm_avg = ri(hc_norm_avg)

plt.figure()
plt.grid()
plt.yscale('log')
plt.title('Sediment transport value (K) comparsion')
plt.plot(hollow_ang, ri_norm, label = '$K_{\mathrm{lin}}$')
plt.plot(hollow_ang, ri_norm_avg, label = '$K_{\mathrm{avg}}$')
plt.legend()
plt.xlabel('$θ_H$')
plt.ylabel('Recurrance Interval (yrs)')

