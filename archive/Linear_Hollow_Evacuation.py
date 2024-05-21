# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Variables for Excavation model 
#infill_rate = 2  # in meters per yr
hollow_length = 10 #in meters
#hollow_width = 8 # in meters

#total_volume = infill rate * hollow_depth *hollow_width

#find critical thickness
C = 2000 #Pa
pw = 1000 # kg/m3
ps = 2000 # kg/m3
phi = np.deg2rad(33)  #converts phi to radians 
g = 9.8  # m/s2
angle = 30
theta = np.deg2rad(angle) # converts theta to radians
Sc = 1.27
K = 0.0036 #m2/yr, this was found in Roering 1999
Qout = 0.0005  #m2/yr
P = 0.00007  #m/yr
width = 8


z_crit_shallow = C / ((pw*g**(np.tan(phi))*(np.cos(theta)))
                      +(ps*g*(np.cos(theta))*(np.tan(theta) - np.tan(phi))))


h = C / (((ps*g**(np.cos(theta))*(np.sin(theta)))) - (ps*g*((np.cos(theta))** 2) * (1 - (pw/ps)*1) * np.tan(phi)))
 
    
print(z_crit_shallow)

# Calculate the qs for a given slope
angle_shallow_side = 36
dz_dx_shallow = np.tan(np.deg2rad(angle_shallow_side))
Qin_shallow = ((K * dz_dx_shallow)/(1-(dz_dx_shallow /Sc)**2))   #m2/yr


## Calculate the residence time for shallow hillslopes
time_shallow = (z_crit_shallow * width) / (2*Qin_shallow - Qout + (P)) ##yrs
print(time_shallow)

print('the recurrance time for hollow failure at a critical soil thickness of ' 
      + str(z_crit_shallow) + ' is ' + str(time_shallow))



    ##calculate the volume
volume_shallow = z_crit_shallow * width * hollow_length * time_shallow
print(volume_shallow)



###############################################################################
## Now for steep hillslopes

#find critical thickness
angle = 42
theta = np.deg2rad(angle) # converts theta to radians

z_crit_steep = C / ((pw*g**(np.tan(phi))*(np.cos(theta)))+(ps*g*(np.cos(theta))*(np.tan(theta) - np.tan(phi))))
print(z_crit_steep)

#Calculate the Qs for steep slopes 

angle_steep_side = 48
dz_dx_steep = np.tan(np.deg2rad(angle))
Qin_steep = ((K * dz_dx_steep)/(1-(dz_dx_steep /Sc)**2))   #m2/yr


## Given certain Qin values, what is the residence time of this hollow

time_steep = (z_crit_steep * width) / (2* Qin_steep - Qout + (P))
print(time_steep)

print('the recurrance time for hollow failure at a critical soil thickness of ' 
      + str(z_crit_steep) + ' is ' + str(time_steep))

## calculate the total volume
volume_steep = z_crit_steep * width * hollow_length * time_steep
print(volume_steep)
























