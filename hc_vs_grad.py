# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:42:00 2022

@author: Selina Davila Olivera
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Calculate critical soil depth for a range of slopes
C = 2000 #Pa
pw = 1000 # kg/m3
ps = 2000 # kg/m3
g = 9.8  # m/s2
phi = np.deg2rad(33)

#establish an array of slope angles
slope = np.array(np.deg2rad(range(1,61)))

# create an array for critical soil depth at angles 1-60 
z_crit = []
for k in slope:
    
    z_value = C / ((pw*g**(np.tan(phi))*(np.cos(k)))
                          +(ps*g*(np.cos(k))*(np.tan(k) 
                                                  - np.tan(phi))))
    z_crit += [z_value]

## Create a dataframe to store these values
hollow_data = pd.DataFrame(zip(slope, z_crit))

# rename column names
hollow_data.columns = ['gradient', 'critical_soil_depth']


## Decrease the cohesion by half
C_dec = 1000 #Pa

z_crit_dec = []
for k in slope:
    z_value = C_dec / ((pw*g**(np.tan(phi))*(np.cos(k)))
                          +(ps*g*(np.cos(k))*(np.tan(k) 
                                                  - np.tan(phi))))
    z_crit_dec += [z_value]

## insert array into dataframe at column 2 positioin 
hollow_data.insert(loc= 2,
          column='critical_soil_depth_dec',
          value=z_crit_dec)
## Increase cohesion to 4000 Pa

C_inc = 4000 #Pa

z_crit_inc = []
for k in slope:
    z_value = C_inc / ((pw*g**(np.tan(phi))*(np.cos(k)))
                          +(ps*g*(np.cos(k))*(np.tan(k) 
                                                  - np.tan(phi))))
    z_crit_inc += [z_value]


hollow_data.insert(loc= 2,
          column='critical_soil_depth_inc',
          value=z_crit_inc)

hollow_data = hollow_data.drop(index = hollow_data.index[40:60])
hollow_data = hollow_data.drop(index = hollow_data.index[0:23])

degrees = np.rad2deg(np.arctan(hollow_data.gradient))

# now make a quick plot about this
#plt.figure()  #this is an empty canvas
plt.figure()
plt.title("Critical Soil Depth at varying Cohesions")
plt.plot(degrees, hollow_data.critical_soil_depth_inc, label = 'Cohesion = 4000 Pa')
plt.plot(degrees, hollow_data.critical_soil_depth, label = 'Cohesion = 2000 Pa')
plt.plot(degrees, hollow_data.critical_soil_depth_dec, label = 'Cohesion = 1000 Pa')

plt.legend()
plt.xlabel('Slope (Degrees)')
plt.ylabel('Depth (m)')
plt.grid()
plt.show()

hollow_data.gradient
hollow_data.critical_soil_depth
