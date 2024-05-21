# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:02:28 2024

@author: sdavilao

Function list for Hollow Evacuation Model
"""

#%%
import numpy as np

#%% Define varibales

# Knon variables
Klin = 0.004 # m2/yr # linear sediment transport coefficient
Sc = 1.25 # unitless # critical slope

# Critical Depth variables
pw = 1000 # kg/m3 # density of water
ps = 1600 # kg/m3c # density of soil
phi = np.deg2rad(41)  # converts phi to radians ## set to 42 right now
g = 9.81  # m/s2 # force of gravity

#Define side/hollow slope range
slope_ang = np.arange(27,51, 0.1) # Side slope range from 27 to 51 in degrees in 0.1 intervals
slope_rad = np.deg2rad(slope_ang) # Side slope in radians
hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang)))))) # Hollow slope in radians
hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang)))))) # Hollow slope in degrees

#%% Function List

# Define a funtion to find Knon given a Klin
# This can be used for any Klin value
def K(a):
    b = a / (1 - ((np.tan(slope_rad)/Sc)**2))
    return b
Knon = K(Klin)

# Define a function to find critical depth for a given cohesion (a)
# This is substituted for cohesion in the Oregon Coast Range (OCR) 
def crit_depth(a):
    result = (a)/(ps*g*(np.cos((hollow_rad))**2)*(np.tan((hollow_rad))-((1-(1*(pw/ps)))*np.tan((phi)))))
    return result

# Define a function for RI given hc
def ri(a):
    B = (((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))
    A = (a**2)/ ((B**2) * (2*(Knon)))
    return A

# Define a function for hollow depth over a specified time of infilling (t) over a range of hollow slopes
def hollow_depth(t):
    result = ((2*Knon)**0.5)*(((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))*(t**0.5)
    return result 