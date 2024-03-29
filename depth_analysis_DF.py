# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:32:07 2023

@author: Selina Davila Olivera

This codes provides a plot of hollow depth as a function of time for a hollow 
of 43 degrees as observed in the field with a 45 degree side slope
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

C = 5600 #kPa ## I can vary this between 5kPa to 9kPa
pw = 1000 # kg/m3
ps = 1600 # kg/m3c
phi = np.deg2rad(41)  # converts phi to radians ## set to 42 right now
g = 9.81  # m/s2
K_avg = 0.004 # m2/yr
Sc = 1.25
slope_ang = np.arange(27,51,0.1)
slope_rad = np.deg2rad(slope_ang)
hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
hollow_ang = np.rad2deg(np.arctan((0.89*(np.tan(np.deg2rad(slope_ang))))))
slope = np.array(np.tan(np.deg2rad(np.arange(27,51,0.1))))
length = 10 #m

def K(a):
    b = a / (1 - ((np.tan(slope_rad)/Sc)**2))
    return b

Kh = K(K_avg)

def hollow_depth(t):
    result = ((2*Kh)**0.5)*(((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))*(t**0.5)
    return result 

t_10 = hollow_depth(10)
t_21 = hollow_depth(21)
t_27 = hollow_depth(27)
t_35 = hollow_depth(35)
t_59 = hollow_depth(59)
t_100 = hollow_depth(100)
t_200 = hollow_depth(200)
t_500 = hollow_depth(500)

df = pd.DataFrame(zip(hollow_ang, slope_ang, Kh, t_10, t_21, t_27, t_35, t_59, t_100, t_200, t_500))
df.columns = ['hollow_angle', 'slope_angle', 'Kh', 'Depth_10','Depth_21','Depth_27','Depth_35','Depth_59','Depth_100','Depth_200','Depth_500']


plt.figure()
# plt.plot(df.hollow_angle, df.Depth_500, label = '500 years')
# plt.plot(df.hollow_angle, df.Depth_200, label = '200 years')
plt.plot(df.hollow_angle, df.Depth_100, label = '100 years', zorder = 1, color = 'green')
#plt.plot(df.hollow_angle, df.Depth_59, label = '59 years', zorder = 1)
plt.plot(df.hollow_angle, df.Depth_27, label = '27 years', zorder = 1, color = 'orange')
plt.plot(df.hollow_angle, df.Depth_21, label = '21 years', zorder = 1, color = 'blue')
plt.scatter(34, 0.27, color = 'orange', label = 'Elliott State Forest')
plt.scatter(30, 0.36, color = 'orange')
plt.scatter(28, 0.27, color = 'orange')
plt.scatter(29, 0.33, color = 'orange')
plt.scatter(30, 0.19, color = 'orange')
plt.scatter(34, 0.16, color = 'orange')
plt.scatter(39, 0.35, color = 'blue', label = 'Hadsall Creek')
#plt.plot(df.hollow_angle, df.Depth_10, label = '10 years')
plt.title('Model Depth Validation')
plt.xlabel('$θ_H$')
plt.ylabel('Depth (m)')
plt.legend()




















