# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:12:57 2023

@author: Selina Davila Olivera
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

K = 0.003 ## m2/yr ## from Roering 1999 this is also Dc in D'Odorico
Dc = 0.0032 ## m2
hollow_length = 10 ## m ## set by Josh
width = 8  ## m ## also set by Josh
C = 2000 #Pa
pw = 1000 # kg/m3
ps = 2000 # kg/m3c
phi = np.deg2rad(33)  #converts phi to radians 
g = 9.8  # m/s2
theta = np.deg2rad(30) # converts theta to radians
theta_steep = np.deg2rad(43) # converts theta to radians


## find K in D'odorico forumla 
hol_ang = np.deg2rad(30*0.8)
K_tri = 2* Dc * np.cos(hol_ang)*((np.tan(theta) ** 2) - (np.tan(hol_ang) ** 2))

## Create an array for time in years
time = np.arange(0, 1500, 2)

#dvdt = K * hollow_length ## in m3/yr

## h equation for triangular hollow geometry D'Odorico as a function of time 
z = []
for i in time:
    h = (K_tri * i)**(0.5)
    z += [h]

## calculate volume over time

volume = []
for i, k in zip(z, time):
    #vol = i * width * hollow_length * k
    vol = ((0.5*i * width) * hollow_length)
    volume += [vol]

## dh/dt is the rate of colluvium accretion
## this decreases with depth, h, of the hollow

dhdt = []
for i in z:
    rate = K_tri/(2*i)
    dhdt += [rate]
    
## find hc for two slopes
## 35 for shallow
## 43 for steep

hc_low = C / ((pw*g**(np.tan(phi))*(np.cos(theta)))
                      +(ps*g*(np.cos(theta))*(np.tan(theta) - np.tan(phi))))

hc_steep = C / ((pw*g**(np.tan(phi))*(np.cos(theta_steep)))
                      +(ps*g*(np.cos(theta_steep))*(np.tan(theta_steep) - np.tan(phi))))
    
## plot
plt.figure()
plt.title("Fill Rate Over Time")
plt.plot(time, dhdt, label = "Fill Rate Over Time")

plt.figure()
plt.title('Hollow Height vs Infill Rate')
plt.plot(z, dhdt, label = "Hollow height vs infill rate")

plt.figure()
plt.xlabel('Time (yrs)')
plt.ylabel('Hollow Depth (m)')
plt.title('Hollow Height over Time')
plt.plot(time, z)
plt.hlines(y= [hc_low], xmin = 0, xmax = 1500, linestyle = 'dashed', color = 'r', lw= 2, label = "Critical Depth for Steep Hollows")
plt.hlines(y= [hc_steep], xmin = 0, xmax = 1500, linestyle = 'dashed', color = 'y', lw= 2, label = "Critical Depth for Steep Hollows")
plt.legend()


plt.figure()
plt.subplot(2,1,1)
plt.plot(time, z)
plt.xlabel('Time (yrs)')
plt.ylabel('Hollow Depth (m)')

plt.subplot(2,1,2)
plt.plot(time, volume)
plt.xlabel('Time (yrs)')
plt.ylabel('Hollow Volume (m3)')