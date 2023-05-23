# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:33:31 2023

@author: sdavilao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## exponential difusion
## establish variables

K_avg = 0.0032 # m2/yr
Sc = 1.27
slope_ang = np.arange(27,48, 0.1)
hollow_ang = 0.8 * slope_ang
slope = np.array(np.tan(np.deg2rad(np.arange(27,48, 0.1))))
hollow_slope = 0.8 * slope
length = 10 #m

## Calculate new K values for exponential diffusion
## Double check this
Kh = []
for i in slope:
    k = K_avg / (1 - ((i/Sc)**2))
    Kh += [k]

## Calculate new Qin variables with no Kh values for different slopes
## this is from slope angles on 1 - 50 degrees

## Q normal for 23 years (tr = 20, tf = 3)
Qin_long = []
for s,k in zip(slope, Kh):
        qs = ((k * s)/(1-(s /Sc)**2))
        Qin_long += [qs]

Q_calcs = pd.DataFrame(zip(slope_ang, hollow_slope, Qin_long))

area_long = []
for k in Qin_long:
    a = k * 23
    area_long += [a]
    
Q_calcs.insert(loc= 3, 
          column='area_long',
          value=area_long)

## Q_fire for 3 years
K_fire = 0.011 ## m2/yr
Sc_fire = 1.03

Qin_fire = []
for s in slope:
        qs = ((K_fire * s)/(1-(s /Sc)**2))
        Qin_fire += [qs]

Q_calcs.insert(loc= 4, 
          column='Qin_fire',
          value=Qin_fire)

area_fire = []
for k in Qin_fire:
    a = k * 3
    area_fire += [a]
    
Q_calcs.insert(loc= 5, 
          column='area_fire',
          value=area_fire)

Qin_norm = []
for i, k in zip(area_long, area_fire):
    q = (i - k)/20      ## 20 for norm years
    Qin_norm += [q]

Q_calcs.insert(loc= 6, 
          column='Qin_norm',
          value=Qin_norm)