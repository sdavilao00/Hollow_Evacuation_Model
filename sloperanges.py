# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:47:17 2022

@author: 12092
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


############# Calculate critical soil depth for a range of slopes ############
C = 2000 #Pa
pw = 1000 # kg/m3
ps = 2000 # kg/m3
g = 9.8  # m/s2
phi = np.deg2rad(33)

#establish an array of slope angles
slope = np.array(np.deg2rad(range(1,36)))

# create an array for critical soil depth at angles 1-60 
z_crit = []
for k in slope:
    
    z_value = C / ((pw*g**(np.tan(phi))*(np.cos(k)))
                          +(ps*g*(np.cos(k))*(np.tan(k) 
                                                  - np.tan(phi))))
    z_crit += [z_value]
############################## Calculate exponention influx ################
Qout = 0.0005  #m2/yr
P = 0.0001  #m/yr
width = 8 # m

# exponential difusion
K_avg = 0.0036 # m2/yr
Sc = 1.27
#dzdx_exp_shallow = np.tan(np.deg2rad(36))

#establish an array of slope angles
slope = np.array(np.deg2rad(range(1,36)))

# create an array for critical soil depth at angles 1-35 
qexp = []
for k in slope:
    qs_value = ((K_avg * k)/(1-(k /Sc)**2))
    qexp += [qs_value]
############################## CALCULATE RES TIME ############################
residence_time = []
for i, j in zip(qexp, z_crit):
    time = j / (i - Qout + (P*width)) 
    residence_time += [time]
    

################## Create a dataframe to store these values ##################
ResTime = pd.DataFrame(zip(slope, z_crit, qexp, residence_time))
ResTime.columns = ['slope', 'critical_depth', 'exponential_influx', 'residence_time']

ResTime = ResTime.drop(index = ResTime.index[0:23])

degrees = np.rad2deg(np.arctan(ResTime.slope))

############################# calculate the volume ############################
length = 10 #in meters

volume= [] 
for k in ResTime.critical_depth:
    vol = k * width * length
    volume += [vol]
  
ResTime.insert(loc = 4,
          column='volume',
          value=volume)   

plt.figure()
plt.plot(degrees, ResTime.residence_time)
plt.xlabel('Slope (Degrees)')
plt.ylabel('Residence time (yrs)')
plt.title('Residence time as a funtion of slope')

plt.figure()
plt.plot(ResTime.residence_time, ResTime.volume)
plt.xlabel('Residence Time (yrs)')
plt.ylabel('Volume (m3)')
plt.title('Volume vs Residence Time')




np.arange(0.0004, 0.0015, 0.00001)









