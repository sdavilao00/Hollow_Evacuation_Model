# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:54:36 2024

@author: sdavilao
"""

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

# Initialize arrays to store values
Knon = np.array([])
hc = np.array([])
ri = np.array([])

# Define a funtion to find Knon given a Klin
# This can be used for any Klin value
def Knon(a):
    b = a / (1 - ((np.tan(slope_rad)/Sc)**2))
    global Knon
    Knon = np.append(Knon, b)  # Save the Knon value to the array
    return b

K = Knon(Klin)

# Define a function to find RI and criticl depth for a given cohesion (a)
# This is substituted for cohesion in the Oregon Coast Range (OCR) 
def ri(a):
    # Nested crit_depth function
    def crit_depth(a):
        result = a / (ps * g * (np.cos(hollow_rad) ** 2) * (np.tan(hollow_rad) - ((1 - (pw / ps)) * np.tan(phi))))
        return result
    
    h = crit_depth(a)
    global hc
    hc = np.append(hc, h)  # Save the crit_depth value to the list
    B = (np.cos(hollow_rad) ** 0.5) * ((np.tan(slope_rad) ** 2 - np.tan(hollow_rad) ** 2) ** 0.25) * ((np.cos(hollow_rad) ** 2 / np.cos(slope_rad) ** 2 - 1) ** 0.25)
    A = (h ** 2) / (B ** 2 * 2 * K)
    global ri
    ri = np.append(ri, A)  # Save the ri value to the array
    return A

# Define a function for hollow depth over a specified time of infilling (t) over a range of hollow slopes
def hollow_depth(t):
    result = ((2*Knon)**0.5)*(((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))*(t**0.5)
    return result 

ri(6800)
