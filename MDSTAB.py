# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:16:26 2024

@author: sdavilao

This code can be used for calculating hollow dimensions along with 
Factor of Safety for a combination of parameters for hollows in the 
Oregon Coast Range. 

This work is based on the three-dimensional slope stability model derived by Milledge et al. 2014. 
    
    
"""
#%% Import Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Variables

# Knon variables
Klin = 0.004 # m2/yr # linear sediment transport coefficient
Sc = 1.25 # unitless # critical slope

# Critical Depth variables
pw = 1000 # kg/m3 # density of water
ps = 1600 # kg/m3c # density of soil
g = 9.81  # m/s2 # force of gravity
yw = g*pw
ys = g*ps
phi = np.deg2rad(41)  # converts phi to radians ## set to 42 right now
#z = np.arange(0,5,0.1)
z = 5

# Slope stability variables
m = 1 # m # saturation ration (h/z)
l = 15 # m # length
w = 7 # m # width
C0 = 6.8 # kPa
j = 4.96

#Define side/hollow slope range
slope_ang = np.arange(27,51, 0.1) # Side slope range from 27 to 51 in degrees in 0.1 intervals
slope_rad = np.deg2rad(slope_ang) # Side slope in radians
hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang)))))) # Hollow slope in radians
hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang)))))) # Hollow slope in degrees

#%% Calculated variables for MDSTAB

# Earth Pressure Variables
K0 = 1-(np.sin(hollow_rad))
Ka = 0.1
Kp = 4

#Cohesion variables
Crb = C0*2.718281**(-z*j)
Crl = (C0/(j*z))*(1 - 2.718281**(-z*j))
# Crb = 5 # kPa # Cohesion of roots on base
# Crl = 5 # kPa # Lateral root cohesion

#%% MDSTAB

#Basal resistence force of the slide block
Frb = (Crb + ((np.cos(hollow_rad))**2)*z*((ys-yw)*m)*np.tan(phi))*l*w

# Resisting force of each cross slope side of the slide block (2 lateral margins)
Frc = (Crl + (K0*0.5*z*((ys-yw)*m**2)*np.tan(phi)))*(np.cos(hollow_rad)*z*l*2)

# Slope parallel component of passive force - slope-parallel component of active force
Frddu = (Kp-Ka)*0.5*z**2*(ys-yw*m**2)*w*((np.cos((2*phi) - hollow_rad))/(np.cos(phi)))

# Central block driving force
Fdc = (np.sin(hollow_rad))*(np.cos(hollow_rad))*z*ps*l*w

#Factor of Safety calculation
FS = (Frb + Frc + Frddu)/Fdc

#%% Visualize

plt.figure()
plt.scatter(hollow_ang, FS)
plt.xlabel('Hollow Angle')
plt.ylabel('Factor of Safety')
