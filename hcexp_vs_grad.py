# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:31:58 2022

@author: Selina Davila Olivera
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###### ADD IN Z CRIT HERE###########################################

# exponential difusion

K_avg = 0.004 # m2/yr
Sc = 1.27
#dzdx_exp_shallow = np.tan(np.deg2rad(36))

#establish an array of slope angles

slope_ang = np.arange(1,51,0.1)
slope_rad = np.deg2rad(slope_ang)
hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))
hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang))))))

# create an array for critical soil depth at angles 1-60 
qexp = []
for k in slope_rad:
    qs_value = ((K_avg * k)/(1-(k /Sc)**2))
    qexp += [qs_value]

## Create a dataframe to store these values
Qin_data = pd.DataFrame(zip(slope_ang, qexp))

# rename column names
Qin_data.columns = ['gradient', 'Qin']



## Linear diffusion 
qlin = []
for k in slope_rad:
    qs_value = (K_avg * k)
    qlin += [qs_value]
    
## Create a dataframe to store these values
Qin_lin = pd.DataFrame(zip(slope_ang, qlin))

# rename column names
Qin_lin.columns = ['gradient', 'Qin_linear']

plt.figure()
plt.plot(Qin_data.gradient, Qin_data.Qin, label = "Nonlinear transport model")
plt.plot(Qin_lin.gradient, Qin_lin.Qin_linear, label = 'Linear transport model', c = 'orange')
plt.legend()
plt.xlabel('Side slope')
plt.ylabel('Sediment Influx')
###############################################################################
# ################ New K values for  exp diffusion
# slope_ang = np.arange(1,51, 0.05)
# slope = np.array(np.tan(np.deg2rad(np.arange(1,51, 0.05))))
# hollow_slope = 0.8 * slope

# Kh = []
# for i in slope:
#     k = K_avg / (1 - ((i/Sc)**2))
#     Kh += [k]

# Qin = []
# for s,k in zip(slope, Kh):
#         qs = ((k * s)/(1-(s /Sc)**2))
#         Qin += [qs]

# # Qout = 0.0005  #m2/yr
# # P = 0.00007  #m/yr     
# # width = 8 #m
# # length = 10 #m

# # dAdt = []
# # for q in Qin:
# #     da = (2*q) - Qout + (P*width)
# #     dAdt += [da]

# volume = pd.DataFrame(zip(slope_ang, slope, Kh, Qin))
# volume.columns = ['slope_deg', 'slope', 'Kh', 'Qin']


# # shallow
# theta_degree = 32  
# theta = np.tan(np.deg2rad(theta_degree))
# hol = 0.8 * theta
# time = np.arange(0,100000, 100)
# #K_val = volume.Kh[(volume.slope_deg)]
# length = 10 #m

# h = []
# for t in time:
#     x = ((2*0.004222119046308766*theta*t)**0.5) * (((((np.tan(hol))**2) - ((np.tan(theta))**2)) * 
#       ((1/((np.cos(hol))**2)) - (1/((np.cos(theta))**2))))**(1/4))
#     h += [x]

# w = []
# for x in h:
#     wid = 2*(x/np.tan(theta))
#     w += [wid]
    
# vol = []
# for i, k in zip(h, w):
#     v = 0.5 * i * k * length
#     vol += [v]

# huh = pd.DataFrame(zip(h, time))

# C = 6000 #Pa
# pw = 1000 # kg/m3
# ps = 2000 # kg/m3c
# phi = np.deg2rad(42)  #converts phi to radians 
# g = 9.8  # m/s2

# hc_low = C / ((pw*g**(np.tan(phi))*(np.cos(theta)))
#                       +(ps*g*(np.cos(theta))*(np.tan(theta) - np.tan(phi))))

# width_low = 2 * (hc_low/(np.tan(theta)))

# volume_low = 0.5 * hc_low * width_low * length

# plt.figure()
# plt.title('Hollow Depth for Slopes at 32 deg')
# plt.plot(time, h)
# plt.hlines(y= [hc_low], xmin = 0, xmax = 5000, linestyle = 'dashed', color = 'r', lw= 2, label = "Critical Depth for Shallow Hollows")
# plt.legend()

# plt.figure()
# plt.title('Hollow Volume for Slopes at 32 deg')
# plt.plot(time, vol)
# plt.hlines(y= [volume_low], xmin = 0, xmax = 5000, linestyle = 'dashed', color = 'r', lw= 2, label = "Critical volume for Shallow Hollows")
# plt.legend()

# # steep
# theta_degree = 43  
# theta = np.tan(np.deg2rad(theta_degree))
# hol = 0.8 * theta
# time = np.arange(0,1001)
# #K_val = float(volume.Kh[(volume.slope_deg == theta_degree)])

# h = []
# for t in time:
#     x = ((2*0.00694359*theta*t)**0.5) * (((((np.tan(hol))**2) - ((np.tan(theta))**2)) * 
#       ((1/((np.cos(hol))**2)) - (1/((np.cos(theta))**2))))**(1/4))
#     h += [x]

# w = []
# for x in h:
#     wid = 2*(x/np.tan(theta))
#     w += [wid]
    
# vol = []
# for i, k in zip(h, w):
#     v = 0.5 * i * k * length
#     vol += [v]



# C = 6000 #Pa
# pw = 1000 # kg/m3
# ps = 2000 # kg/m3c
# phi = np.deg2rad(42)  #converts phi to radians 
# g = 9.8  # m/s2

# hc_high = C / ((pw*g**(np.tan(phi))*(np.cos(theta)))
#                       +(ps*g*(np.cos(theta))*(np.tan(theta) - np.tan(phi))))

# width_high = 2 * (hc_low/(np.tan(theta)))

# volume_high = 0.5 * hc_low * width_low * length

# plt.figure()
# plt.title('Hollow Depth for Slopes at 43 deg')
# plt.plot(time, h)
# plt.hlines(y= [hc_high], xmin = 0, xmax = 1000, linestyle = 'dashed', color = 'r', lw= 2, label = "Critical Depth for Steep Hollows")
# plt.legend()

# plt.figure()
# plt.title('Hollow volume for Slopes at 43 deg')
# plt.plot(time, vol)
# plt.hlines(y= [volume_low], xmin = 0, xmax = 1000, linestyle = 'dashed', color = 'r', lw= 2, label = "Critical volume for Steep Hollows")
# plt.legend()

