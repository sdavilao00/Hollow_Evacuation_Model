# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:32:30 2024

@author: sdavilao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## exponential difusion
## establish variables
C = 6800 #kPa ## I can vary this between 5kPa to 9kPa
pw = 1000 # kg/m3
ps = 1600 # kg/m3c
phi = np.deg2rad(41)  # converts phi to radians ## set to 42 right now
g = 9.81  # m/s2
K_avg = 0.004 # m2/yr
Sc = 1.25
slope_ang = np.arange(27,51,0.1)
slope_rad = np.deg2rad(slope_ang)
hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
# slope = np.array(np.tan(np.deg2rad(np.arange(27,51,0.1))))
# length = 10 #m