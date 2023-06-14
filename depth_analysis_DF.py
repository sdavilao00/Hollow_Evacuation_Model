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

#38.65
hollow_rad = np.deg2rad(43)
slope_rad = np.deg2rad(45)
Kh = 0.0111111
t = np.arange(0, 100000, 2)

h_t = []
for i in t:
    h = ((2*Kh)**0.5)*(((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))*(i**0.5)
    h_t += [h]
    
    
df = pd.DataFrame(zip(t, h_t))