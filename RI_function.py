# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:30:21 2023

@author: Selina Davila Olivera

This code can be used to calculate: 
    
1) Slope-dependent nonlinear sediment transport coefficients (knon) for a given 
    linear sediment transport coefficient value (klin) over a range of slopes
    -- Defined field derived varibles below (Roering et al. 1999)

2) Critical depth (using a one dimensional infitie slope model) as a function 
    of field derived varibales for the OCR (Schroeder, 1983), and
    cohesion (values derived by Schmidt et al. 2001)

3) Recurrnce Intervals for each calculated critical depth for a given cohesion
    (Derived by Dietrich et al. 1986, modified), over a range of slopes,
    and invoking a slope dependent nonlinear sediment transport coefficient

"""
#%%
# Import needed packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Define varibales

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

#%% 
# Define function for nonlinear sediment transport coefficient (knon), 
# critical depth (hc) and Recurrence Interval (RI)

# Define a funtion to find Knon given a Klin
# This can be used for any Klin value
def K(a):
    b = a / (1 - ((np.tan(slope_rad)/Sc)**2))
    return b

#Solve for Knon for a range of slopes given Klin of 0.004 m2/yr
Knon = K(Klin)

#%%
# Define a function to find critical depth for a given cohesion (a)
# This is substituted for cohesion in the Oregon Coast Range (OCR) 
def crit_depth(a):
    result = (a)/(ps*g*(np.cos((hollow_rad))**2)*(np.tan((hollow_rad))-((1-(1*(pw/ps)))*np.tan((phi)))))
    return result

# Find critical depth for different cohesions in the OCR
hc_if = crit_depth(6800) #critical height for 6800 Pa - Industiral Forest
hc_ccf = crit_depth(2500) #critical height for 2500 Pa - Clear Cut/Fire
hc_dog = crit_depth(25600) #critical height for 25600 Pa - Dense Old Growth

#%%
# Define a function for RI given hc
def ri(a):
    B = (((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))
    A = (a**2)/ ((B**2) * (2*(Knon)))
    return A

# Find RI for each hc
ri_if = ri(hc_if) # RI for 6800 kPa
ri_ccf = ri(hc_ccf) # RI for 2500 kPa
ri_dog = ri(hc_dog) # RI for 25600 kPa

#%% Plot data
#### RI comparison for different cohesions over range of hollow slopes #######

# Set Seaborn style
sns.set(style='darkgrid')

# Create the plot
plt.figure(dpi = 150) # this is mainly for exporting purposes but not necessary
plt.title("Cohesion Comparison", fontsize = 16)
plt.yscale('log')

# Plot lines
sns.lineplot(x=hollow_ang, y=ri_dog, label='C = 25.6 kPa - Dense Old Growth')
sns.lineplot(x=hollow_ang, y=ri_if, label='C = 6.8 kPa - Industrial Forest')
sns.lineplot(x=hollow_ang, y=ri_ccf, label='C = 2.5 kPa - Clear Cut/Fire')

# Add labels and legend
plt.legend()
plt.xlabel('$θ_H$', fontsize = 14)
plt.ylabel('Recurrence Interval (yrs)', fontsize = 14)

# Show plot
plt.show()

#%% 
## RI comparison for Nonlinear K and Linear K over range of hollow slopes #####

# Find a RI value using a linear sediment transport coefficent (Klin)
# Set a desired cohesion value of 6.8 kPa
# Find the critical depth for this cohesion 
hc_if_lin = crit_depth(6800)

# Run the Recurrence Interval function for this critical height with Klin
ri_if_lin = ri(hc_if_lin)

# Plot the result using seaborn
# Set Seaborn style
sns.set(style='darkgrid')

# Create the plot
plt.figure(dpi = 150) # this is mainly for exporting purposes but not necessary
plt.title('Sediment transport value (K) comparison', fontsize = 16)
plt.yscale('log')

# Plot lines
sns.lineplot(x=hollow_ang, y=ri_if, label='$K_{\mathrm{non}}$')
sns.lineplot(x=hollow_ang, y=ri_if_lin, label='$K_{\mathrm{lin}}$')

# Add labels and legend
plt.legend()
plt.xlabel('$θ_H$', fontsize = 14)
plt.ylabel('Recurrance Interval (yrs)', fontsize = 14)

# Show plot
plt.show()



plt.figure()
plt.plot(hollow_ang, hc_if)
plt.plot(hollow_ang, hc_ccf)