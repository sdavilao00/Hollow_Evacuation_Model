# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:47:33 2023

@author: 12092
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
slope = np.array(np.tan(np.deg2rad(np.arange(27,51,0.1))))
length = 10 #m

## Calculate new K values for exponential diffusion
## Double check this
Kh = []
for i in slope_rad:
    k = K_avg / (1 - ((np.tan(i)/Sc)**2))
    Kh += [k]

## Calculate new Qin variables with no Kh values for different slopes
## this is from slope angles on 1 - 50 degrees

Qin_norm = []
for s,k in zip(slope, Kh):
        qs = ((k * s)/(1-(s /Sc)**2))
        Qin_norm += [qs]

hc_norm = []
for k in hollow_rad:
    h = (C)/(ps*g*(np.cos((k))**2)*(np.tan((k))-((1-(1*(pw/ps)))*np.tan((phi)))))
    hc_norm += [h]
   
w_norm = []
for x,s in zip(hc_norm, slope_rad):
    wid = 2*(x/np.tan(s))
    w_norm += [wid]

volume_norm = []
for h, x in zip(hc_norm, w_norm):
    v = 0.5 * x * h * length
    volume_norm += [v]
    
ri = []
for v, i, z, k in zip(hc_norm, Kh, slope_rad, hollow_rad):
    B = (((np.cos(k))**(1/2))*(((np.tan(z))**2)-((np.tan(k))**2))**(1/4)) * ((((np.cos(k)**2)/(np.cos(z)**2))-1)**(1/4))
    ## take out 1/2
    y = (v**2)/ ((B**2) * (2*i))
    ri += [y]

df = pd.DataFrame(zip(slope_ang, hollow_ang, Kh, hc_norm, volume_norm, ri))
### low cohesion
C = 2700 #Pa ## I can vary this between 5kPa to 9kPa

## Calculate new Qin variables with no Kh values for different slopes
## this is from slope angles on 1 - 50 degrees

Qin_low = []
for s,k in zip(slope, Kh):
        qs = ((k * s)/(1-(s /Sc)**2))
        Qin_low += [qs]

hc_low = []
for k in hollow_rad:
    h = (C)/(ps*g*(np.cos((k))**2)*(np.tan((k))-((1-(1*(pw/ps)))*np.tan((phi)))))
    hc_low += [h]
   
w_low = []
for x,s in zip(hc_low, slope_rad):
    wid = 2*(x/np.tan(s))
    w_low += [wid]

volume_low = []
for h, x in zip(hc_low, w_low):
    v = 0.5 * x * h * length
    volume_low += [v]
    
ri_low = []
for v, i, z, k in zip(hc_low, Kh, slope_rad, hollow_rad):
    B = (((np.cos(k))**(1/2))*(((np.tan(z))**2)-((np.tan(k))**2))**(1/4)) * ((((np.cos(k)**2)/(np.cos(z)**2))-1)**(1/4))
    ## take out 1/2
    y = (v**2)/ ((B**2) * (2*i))
    ri_low += [y]

## Medium
C = 4500 #Pa ## I can vary this between 5kPa to 9kPa

## Calculate new Qin variables with no Kh values for different slopes
## this is from slope angles on 1 - 50 degrees

Qin_med = []
for s,k in zip(slope, Kh):
        qs = ((k * s)/(1-(s /Sc)**2))
        Qin_med += [qs]

hc_med = []
for k in hollow_rad:
    h = (C)/(ps*g*(np.cos((k))**2)*(np.tan((k))-((1-(1*(pw/ps)))*np.tan((phi)))))
    hc_med += [h]
   
w_med = []
for x,s in zip(hc_med, slope_rad):
    wid = 2*(x/np.tan(s))
    w_med += [wid]

volume_med = []
for h, x in zip(hc_med, w_med):
    v = 0.5 * x * h * length
    volume_med += [v]
    
ri_med = []
for v, i, z, k in zip(hc_med, Kh, slope_rad, hollow_rad):
    B = (((np.cos(k))**(1/2))*(((np.tan(z))**2)-((np.tan(k))**2))**(1/4)) * ((((np.cos(k)**2)/(np.cos(z)**2))-1)**(1/4))
    ## take out 1/2
    y = (v**2)/ ((B**2) * (2*i))
    ri_med += [y]

## high C
C = 11000 #Pa ## I can vary this between 5kPa to 9kPa

## Calculate new Qin variables with no Kh values for different slopes
## this is from slope angles on 1 - 50 degrees

Qin_high = []
for s,k in zip(slope, Kh):
        qs = ((k * s)/(1-(s /Sc)**2))
        Qin_high += [qs]

hc_high = []
for k in hollow_rad:
    h = (C)/(ps*g*(np.cos((k))**2)*(np.tan((k))-((1-(1*(pw/ps)))*np.tan((phi)))))
    hc_high += [h]
   
w_high = []
for x,s in zip(hc_high, slope_rad):
    wid = 2*(x/np.tan(s))
    w_high += [wid]

volume_high = []
for h, x in zip(hc_high, w_high):
    v = 0.5 * x * h * length
    volume_high += [v]
    
ri_high = []
for v, i, z, k in zip(hc_high, Kh, slope_rad, hollow_rad):
    B = (((np.cos(k))**(1/2))*(((np.tan(z))**2)-((np.tan(k))**2))**(1/4)) * ((((np.cos(k)**2)/(np.cos(z)**2))-1)**(1/4))
    ## take out 1/2
    y = (v**2)/ ((B**2) * (2*i))
    ri_high += [y]
    
# max natural forest
C = 25600 #Pa ## I can vary this between 5kPa to 9kPa

Qin_max = []
for s,k in zip(slope, Kh):
        qs = ((k * s)/(1-(s /Sc)**2))
        Qin_max += [qs]

hc_max = []
for k in hollow_rad:
    h = (C)/(ps*g*(np.cos((k))**2)*(np.tan((k))-((1-(1*(pw/ps)))*np.tan((phi)))))
    hc_max += [h]
   
w_max = []
for x,s in zip(hc_max, slope_rad):
    wid = 2*(x/np.tan(s))
    w_max += [wid]

volume_max = []
for h, x in zip(hc_max, w_max):
    v = 0.5 * x * h * length
    volume_max += [v]
    
ri_max = []
for v, i, z, k in zip(hc_max, Kh, slope_rad, hollow_rad):
    B = (((np.cos(k))**(1/2))*(((np.tan(z))**2)-((np.tan(k))**2))**(1/4)) * ((((np.cos(k)**2)/(np.cos(z)**2))-1)**(1/4))
    ## take out 1/2
    y = (v**2)/ ((B**2) * (2*i))
    ri_max += [y] 

plt.figure()
plt.grid()
plt.title("Cohesion Comparison with NonLinear K")
#plt.yscale('log')
plt.plot(hollow_ang, ri_max, label = 'C = 25.6 kPa')
plt.plot(hollow_ang, ri_high, label = 'C = 11 kPa')
plt.plot(hollow_ang, ri, label = 'C = 6.8 kPa')
plt.plot(hollow_ang, ri_med, label = 'C = 4.5 kPa')
plt.plot(hollow_ang, ri_low, label = 'C = 2.7 kPa')
plt.legend()
plt.xlabel('Hollow Slope (degrees)')
plt.ylabel('Recurrance Interval (yrs)')

