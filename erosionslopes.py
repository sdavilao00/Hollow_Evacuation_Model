# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 21:12:24 2022

@author: Selina Davila Olivera
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## zcrit variables 
C = 2000 #Pa
pw = 1000 # kg/m3
ps = 2000 # kg/m3
phi = np.deg2rad(33)  #converts phi to radians 
g = 9.8  # m/s2
## time variables 
P = 0.00007  #m/yr
width = 8  ## m 
hollow_length = 10 ## m ## set by Josh
Qout = 0.0005  #m2/yr
K = 0.003 ## m2/yr ## from Roering 1999
Sc = 1.2 ## unitless (grad) ## from Roering 1999
E = 0.001 ## mm/ yr Roering 2005
## mm/yr ## this is an avergae, erosion rates range from 0.07 to
## 0.15 in the OCR
Lh = 57 ## m ## hollow length, this is from Roering 2005 for the OCR
Dc = 0.0032 ## m2 from D'Odorico



ps = 2000 ## kg/m3 ## Density of soil
pr = 4000 ## kg/m3 ## roering 2005
## Section A of EQN


#slope_angle = np.rad2deg(np.arctan(S))
#hollow_angle = slope_angle * 0.8


## Let's get slope angles for a certain amount of erosion
## I will do this by making a for loop for different E values
## Then pluggin this in to the given slope equation
## From there, I can get critical depth and volumes with this




E_range = np.arange(0.00004, 0.00015, 0.000001)



## Calculate E* for different erosion rates, E* is the dimensionless erosion rate
erosion_star = []
for i in E_range:
    e = (2 * i *(pr/ps)* Lh)/(K* Sc)
    erosion_star += [(e)]
   
##Now calculate slope for these E* values
## This is eqn 10 in Roering 2005


slope_gradient = []
for k in erosion_star:
    s = ((1/k) * (((1 + (k **2)) ** 0.5) - (np.log(0.5*(1 + ((1+(k ** 2))** 0.5))))-1)) * Sc
    slope_gradient += [s]

## this converts slope gradients found in the previous steps degrees 
 
slope_angle = np.rad2deg(np.arctan(slope_gradient))
slope_radian = (np.arctan(slope_gradient))
## find hollow angles
hollow_angle = 0.8 * slope_angle
hollow_radian = 0.8 * slope_radian
        
## Now that I have slope angles, I'll now calculate new critical hollow depths
## From there, I can calculate residence times 
## Then, total volumes

## z critical 
## this may need to be in radians
hc = []
for k in slope_radian:
    h = (C / (((ps*g**(np.cos(k))*(np.sin(k)))) - (ps*g*((np.cos(k))** 2) * (1 - (pw/ps)*1) * np.tan(phi))))*np.cos(k)
    hc += [h]

##We will need to find a new K value for the triangular geometry and creep

#K_tri = []
#for i, k in zip(hollow_radian, slope_radian):
    #k = 2* Dc * np.cos(i)*((np.tan(k) ** 2) - (np.tan(i) ** 2))
    #K_tri += [k]

## influx of sediment
Qin = []
for s in slope_gradient:
        qs = ((K * s)/(1-(s /Sc)**2))
        Qin += [qs]
    
## Calculate the total volume during these slopes

volume = []
for i in hc:
    #vol = i * width * hollow_length * k
    vol = ((0.5 * i * width) * hollow_length)
    volume += [vol]
    
## calculate residence times

res_time = []
for i,k in zip(volume, Qin):
    t = i / ((2*k*hollow_length) - (Qout * hollow_length) + (P*hollow_length*width))
    res_time += [t]
### THIS NEEDS TO BE CHECKED


## Create a pandas datafram for the follow variable above
hollow_volume = pd.DataFrame(zip(E_range, erosion_star, slope_gradient, slope_angle, hc, Qin, res_time, volume))

## Rename columns of dataframe
hollow_volume.columns = ['erosion', 'e_star', 'gradient', 'angle_deg', 'critical_depth', 'influx', 'residence_time', 'volume']


## add in hollow slopes, this is 0.8 of the side slope angles (Deitrich 90)
hollow_angle = hollow_volume.angle_deg * 0.8

## add this in to the dataframe
hollow_volume.insert(loc= 4,
          column='hollow_angle',
          value=hollow_angle)


plt.figure()
plt.plot(hollow_volume.erosion*1000, hollow_volume.residence_time)
plt.scatter(hollow_volume.erosion*1000, hollow_volume.residence_time, c = 'k')
plt.xlabel('Erosion Rate (mm/yr)')
plt.ylabel('Recurrence Interval (yrs)')
plt.title('Recurrence Interval as a function of Erosion Rate')

plt.figure()
plt.plot(hollow_volume.hollow_angle, hollow_volume.residence_time, c = 'k')
plt.xlabel('Slope of Hollow (deg)')
plt.ylabel('Recurrence Interval')
plt.title('Recurrence Interval as a function of Hollow Angle')
## to do next
## change z critical values to formula that includes h/z
#hz = [0, 0.2, 0.4, 0.6, 0.8, 1]

## I either alter the h equation so that h and z are perpendicular to the bedrock surface
## OR alter the odorico equations for h/z ratios

## Dry conditions
hc_dry = []
for k in slope_radian:
    h = (C / (((ps*g**(np.cos(k))*(np.sin(k)))) - (ps*g*((np.cos(k))** 2) * (1 - (pw/ps)*0) * np.tan(phi))))* np.cos(k)
    #h = C / (ps*g*np.cos(k)*(np.tan(k) - np.tan(phi)))
    hc_dry += [h]


## volume for dry condiitons    
volume_dry = []
for i in hc_dry:
    vol = ((0.5 * i * width) * hollow_length)
    volume_dry += [vol]

## residence time for completely dry
res_time_dry = []
for i,k in zip(volume_dry, Qin):
    t = i / ((2*k*hollow_length) - (Qout * hollow_length) + (P*hollow_length*width))
    res_time_dry += [t]
    
############
    
## critical depth for h/z at full saturation
hc_sat = []
for k in slope_radian:
    h = (C / (((ps*g**(np.cos(k))*(np.sin(k)))) - (ps*g*((np.cos(k))** 2) * (1 - (pw/ps)*1) * np.tan(phi))))*np.cos(k)
    hc_sat += [h]
    
## volume for saturation
volume_sat = []
for i in hc_sat:
    vol = ((0.5 * i * width) * hollow_length)
    volume_sat += [vol]

## residence time for full saturation
res_time_sat = []
for i,k in zip(volume_sat, Qin):
    t = i / ((2*k*hollow_length) - (Qout * hollow_length) + (P*hollow_length*width))
    res_time_sat += [t]
    
############

## critical depth for half h/z
hc_sat_50 = []
for k in slope_radian:
    h = (C / (((ps*g**(np.cos(k))*(np.sin(k)))) - (ps*g*((np.cos(k))** 2) * (1 - (pw/ps)*0.5) * np.tan(phi)))) * np.cos(k)
    hc_sat_50 += [h]

volume_50 = []
for i in hc_sat_50:
    vol = ((0.5 * i * width) * hollow_length)
    volume_50 += [vol]


## res time for half saturation
res_time_50 = []
for i,k in zip(volume_50, Qin):
    t = i / ((2*k*hollow_length) - (Qout * hollow_length) + (P*hollow_length*width))
    res_time_50 += [t]
    
#############

## critical depth for quarter saturation
hc_sat_25 = []
for k in slope_radian:
    h = (C / (((ps*g**(np.cos(k))*(np.sin(k)))) - (ps*g*((np.cos(k))** 2) * (1 - (pw/ps)*0.25) * np.tan(phi))))* np.cos(k)
    hc_sat_25 += [h]
  
volume_25 = []
for i in hc_sat_25:
    vol = ((0.5 * i * width) * hollow_length)
    volume_25 += [vol]
    
##  residence time for quarter saturation
res_time_25 = []
for i,k in zip(volume_25, Qin):
    t = i / ((2*k*hollow_length) - (Qout * hollow_length) + (P*hollow_length*width))
    res_time_25 += [t]


    
#############

## critical depth for 3/4 saturation
hc_sat_75 = []
for k in slope_radian:
    h = (C / (((ps*g**(np.cos(k))*(np.sin(k)))) - (ps*g*((np.cos(k))** 2) * (1 - (pw/ps)*0.75) * np.tan(phi)))) * np.cos(k)
    hc_sat_75 += [h]

volume_75 = []
for i in hc_sat_75:
    vol = ((0.5 * i * width) * hollow_length)
    volume_75 += [vol]
    
    
 ## residence time for 3/4 saturation
res_time_75 = []
for i,k in zip(volume_75, Qin):
    t = i / ((2*k*hollow_length) - (Qout * hollow_length) + (P*hollow_length*width))
    res_time_75 += [t]

    
    
## Create a dataframe
restime_sat = pd.DataFrame(zip(hollow_angle, res_time_dry, volume_dry, res_time_sat, volume_sat, res_time_25, volume_25,
                               res_time_50, volume_50, res_time_75, volume_75))
restime_sat.columns = ['hollow_angle', 'res_time_dry', 'volume_dry','res_time_sat', 'volume_sat',
                       'res_time_25', 'volume_25', 'res_time_50', 'volume_50', 'res_time_75', 'volume_75']


## Find the row indicies that don't have negative residence times for each conditions
## then make these into their own df for plotting
indicies0 = np.where(restime_sat.res_time_dry > 0)
indicies25 = np.where(restime_sat.res_time_25 > 0)
indicies50 = np.where(restime_sat.res_time_50 > 0)

## the dataframes
sat_0 = restime_sat.iloc[indicies0]
sat_25 = restime_sat.iloc[indicies25]
sat_50 = restime_sat.iloc[indicies50]

## plot for residence time
plt.figure()
plt.yscale('log')
plt.grid()
plt.title('Recurrence Interval as a function of Hollow Angle at different saturations')
plt.plot(sat_0.hollow_angle, sat_0.res_time_dry, label = "Dry Conditions")
plt.plot(sat_25.hollow_angle, sat_25.res_time_25, label = '25% Saturated')
plt.plot(sat_50.hollow_angle, sat_50.res_time_50, label = '50% Saturated')
plt.plot(restime_sat.hollow_angle, restime_sat.res_time_75, label = '75% Saturated')
plt.plot(restime_sat.hollow_angle, restime_sat.res_time_sat, label = 'Fully Saturated')
#plt.plot(hollow_volume.hollow_angle, hollow_volume.residence_time, label = 'Saturated(DOdorico 2003)')
plt.legend(loc = 'upper left')
plt.xlabel('Hollow Angle (deg)')
plt.ylabel('Recurrence Interval (yrs)')

## plot for volume
plt.figure()
plt.yscale('log')
plt.grid()
plt.title('Hollow Volume as a function of Hollow Angle at different saturations')
plt.plot(sat_0.hollow_angle, sat_0.volume_dry, label = "Dry Conditions")
plt.plot(sat_25.hollow_angle, sat_25.volume_25, label = '25% Saturated')
plt.plot(sat_50.hollow_angle, sat_50.volume_50, label = '50% Saturated')
plt.plot(restime_sat.hollow_angle, restime_sat.volume_75, label = '75% Saturated')
plt.plot(restime_sat.hollow_angle, restime_sat.volume_sat, label = 'Fully Saturated')
#plt.plot(hollow_volume.hollow_angle, hollow_volume.residence_time, label = 'Saturated(DOdorico 2003)')
plt.legend(loc = 'upper left')
plt.xlabel('Hollow Angle (deg)')
plt.ylabel('Volume (m3)')


