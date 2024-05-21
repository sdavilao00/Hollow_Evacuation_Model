# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:42:31 2023

@author: Selina Davila Olivera
"""
## import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## exponential difusion
## establish variables
C = 6000 #Pa ## I can vary this between 5kPa to 9kPa
pw = 1000 # kg/m3
ps = 2000 # kg/m3c
phi = np.deg2rad(42)  #converts phi to radians ## set to 42 right now
g = 9.8  # m/s2
K_avg = 0.0032 # m2/yr
Sc = 1.27
slope_ang = np.arange(27,48, 0.1)
hollow_ang = 0.8 * slope_ang
slope = np.array(np.tan(np.deg2rad(np.arange(27,48,0.1))))
hollow_slope = 0.8 * slope
length = 10 #m
Qout = 0.0005  #m2/yr
P = 0.00007  #m/yr

## Calculate new K values for exponential diffusion
## Double check this
Kh = []
for i in slope:
    k = K_avg / (1 - ((i/Sc)**2))
    Kh += [k]

## Calculate new Qin variables with no Kh values for different slopes
## this is from slope angles on 1 - 50 degrees

Qin_norm = []
for s,k in zip(slope, Kh):
        qs = ((k * s)/(1-(s /Sc)**2))
        Qin_norm += [qs]

hc_norm = []
for k in slope:
    h = C / ((pw*g**(np.tan(phi))*(np.cos(k)))
                          +(ps*g*(np.cos(k))*(np.tan(k) - np.tan(phi))))
    hc_norm += [h]
   
w_norm = []
for x,s in zip(hc_norm, slope):
    wid = 2*(x/np.tan(s))
    w_norm += [wid]

volume_norm = []
for h, x in zip(hc_norm, w_norm):
    v = 0.5 * x * h * length
    volume_norm += [v]

ri = []
for v, i, z, k in zip(hc_norm, Kh, slope, hollow_slope):
    y = ((v/(((((np.tan(k))**2) - ((np.tan(z))**2)) * 
      ((1/((np.cos(k))**2)) - (1/((np.cos(z))**2))))**(1/4)))**2)/ (2*i*z)
    ri += [y]

df = pd.DataFrame(zip(slope_ang, hollow_ang, Kh, Qin_norm, hc_norm, volume_norm, ri))
df.columns = ['slope_ang', 'hollow_ang', 'Kh', 'Qin_norm', 'hc_norm', 'volume_norm', 'ri']


plt.figure()
#plt.grid()
plt.title("Recurrence Interval for varying Hollow Angles")
plt.plot(hollow_ang, ri)
plt.xlabel('Hollow Slope (deg)')
plt.ylabel('Recurrence Interval (yrs)')
#plt.yscale('log')


plt.figure()
plt.plot(hollow_ang, volume_norm)


t = 2600
ri_low = df.loc[slope_ang == 32, 'ri']
t_low = t/ri_low
vol_low = (df.loc[slope_ang == 32, 'volume_norm'])
vollow_tot = t_low * vol_low


ri_high = df.loc[slope_ang == 43, 'ri']
t_high = t/ri_high
vol_high = (df.loc[slope_ang == 43, 'volume_norm'])
volhigh_tot = t_high * vol_high

##############################################################################
######################CONSTANT K##############################################

ri_kavg = []
for v, z, k in zip(hc_norm, slope, hollow_slope):
    y = ((v/(((((np.tan(k))**2) - ((np.tan(z))**2)) * 
      ((1/((np.cos(k))**2)) - (1/((np.cos(z))**2))))**(1/4)))**2)/ (2*K_avg*z)
    ri_kavg += [y]
    
plt.figure()
#plt.grid()
plt.title("Recurrence Interval for varying Hollow Angles")
plt.plot(hollow_ang, ri)
plt.plot(hollow_ang, ri_kavg)
plt.xlabel('Hollow Angle (deg)')
plt.ylabel('Recurrence Interval (yrs)')
#plt.yscale('log')

##############################################################################
######################Linear Qin##############################################


Qin_lin = []
for s in slope:
    qs = (K_avg * s)
    Qin_lin += [qs]

vol_in = []
for q in Qin_lin:
    v = (length * (2 *q))
    vol_in += [v]

ri_lin = []
for i, k in zip(vol_in, volume_norm):
    r = k/i
    ri_lin += [r]


plt.figure()
plt.grid()
plt.title("Recurrence Interval for varying Hollow Angles")
plt.plot(hollow_ang, ri_kavg, label = 'Linear K, Exponential Qin')
plt.plot(hollow_ang, ri, label = 'Exponential K and Qin')
plt.plot(hollow_ang, ri_lin, label = 'Linear K and Qin')
plt.xlabel('Hollow Angle (deg)')
plt.ylabel('Recurrence Interval (yrs)')
plt.yscale('log')
plt.legend()





















