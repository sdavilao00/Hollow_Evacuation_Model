# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:00:59 2023

@author: Selina Davila Olivera
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# exponential difusion without fire

K_avg = 0.0032 # m2/yr
Sc = 1.27
#dzdx_exp_shallow = np.tan(np.deg2rad(36))

#establish an array of slope angles
slope = np.arange(25,45,0.2)
slope_gradient = np.array(np.tan(np.deg2rad(slope)))

## establish a timeframe 
time = np.arange(0, 2000, 2)

#Hollow length
length = 10 #m 

#create an array for critical soil depth at angles 1-60 
Qexp = []
for k in slope_gradient:
    qs_value = ((K_avg * k)/(1-(k /Sc)**2))
    Qexp += [qs_value]

# vol_norm = []
# for i,k in zip(Qexp, time):
#     vol = length * i * k
#     vol_norm += [vol]

####################################################################
##################### FIRE #########################################

## New variables for fire scenario
K_fire = 0.011 # +- 35 cm2/yr - m2/yr, this is from Gerber 2005
Sc_fire = 1.03 # +- 0.2 #also from Gerber 2005

## Influx of sediment in m2/yr     
Qfire = []
for s in slope_gradient:
        qs = ((K_fire * s)/(1-(s/Sc_fire)**2))
        Qfire += [qs]


## Volume over time

# vol_fire = []
# for i,k in zip(Qfire, time):
#     vol = length * 2*i * k
#     vol_fire += [vol]

fire_erosion = pd.DataFrame(zip(slope_gradient, slope, Qexp, Qfire))

plt.figure()
plt.grid()
plt.plot(slope_gradient, Qfire, label = 'Postfire Flux')
plt.plot(slope_gradient, Qexp, label = 'Longterm Flux')
plt.legend()
plt.xlabel('Slope')
plt.ylabel('Flux ($m^2$/yr)')

# #plt.figure()
# plt.plot(time, vol_norm, label = "Volume in normal conditions")
# plt.plot(time, vol_fire, label = "Volume after fire")
# plt.legend()
# plt.xlabel('Time (yrs)')
# plt.ylabel('Volume ($m^3$)')

