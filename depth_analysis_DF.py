# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:32:07 2023

@author: Selina Davila Olivera

In this code you will find:

1) Modeled depth for a given time of infilling over a range of hollow slopes 
    given revamped parameter in the RI_function script
    
2) Modeled depth over time for a specifice hollow angle

*This should be modified so that the Knon and RI time value can be pulled instead of having
to go back in the variable explorer and search for it yourself*
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

#%% Modeled depth for a given time of infilling over a range of hollow slopes
# Define function for nonlinear sediment transport coefficient (knon) 
# and hollow depth

# Define a funtion to find Knon given a Klin
# This can be used for any Klin value
def K(a):
    b = a / (1 - ((np.tan(slope_rad)/Sc)**2))
    return b

#Solve for Knon for a range of slopes given Klin of 0.004 m2/yr
Knon = K(Klin)

# Define a function for hollow depth over a specified time of infilling (t) over a range of hollow slopes
def hollow_depth(t):
    result = ((2*Knon)**0.5)*(((np.cos(hollow_rad))**(1/2))*(((np.tan(slope_rad))**2)-((np.tan(hollow_rad))**2))**(1/4)) * ((((np.cos(hollow_rad)**2)/(np.cos(slope_rad)**2))-1)**(1/4))*(t**0.5)
    return result 

#%%
# Hollow depth for time of infilling
t_10 = hollow_depth(10)
t_20 = hollow_depth(20)
t_27 = hollow_depth(27)
t_35 = hollow_depth(35)
t_50 = hollow_depth(50)
t_100 = hollow_depth(100)
t_200 = hollow_depth(200)
t_500 = hollow_depth(500)
t_1000 = hollow_depth(1000)
t_5000 = hollow_depth(5000)
t_10000 = hollow_depth(10000)
t_50000 = hollow_depth(50000)
t_80000 = hollow_depth(80000)

# Make a dataframe for this
df = pd.DataFrame(zip(hollow_ang, slope_ang, Knon, t_10, t_20, t_27, t_35, t_50, t_100, t_200, t_500, t_1000, t_5000, t_10000, t_50000, t_80000))
df.columns = ['hollow_angle', 'slope_angle', 'Knon', 'Depth_10','Depth_20','Depth_27','Depth_35','Depth_50','Depth_100','Depth_200',
              'Depth_500', 'Depth_1000', 'Depth_5000', 'Depth_10000', 'Depth_50000', 'Depth_80000']

#%% 
# Plot depth for given hollow infilling for given hollow slope
# Set the style
sns.set(style="darkgrid")

# Plot the data
plt.figure(dpi = 150)
sns.lineplot(data=df, x='hollow_angle', y='Depth_100', label = "Reference", color='black')
sns.lineplot(data=df, x='hollow_angle', y='Depth_50', color='black')
sns.lineplot(data=df, x='hollow_angle', y='Depth_20', color='black')

# These are the survey depths for six hollows with their respective error bars
plt.errorbar(34, 0.27, yerr=0.11, fmt='-o', color='orange', linestyle='None', capsize=5, capthick=2, label='Elliott State Forest - 27 years of Infilling')
plt.errorbar(30, 0.36, yerr=0.06, fmt='-o', color='orange', linestyle='None', capsize=5, capthick=2)
plt.errorbar(28, 0.27, yerr=0.05, fmt='-o', color='orange', linestyle='None', capsize=5, capthick=2)
plt.errorbar(29, 0.33, yerr=0.12, fmt='-o', color='orange', linestyle='None', capsize=5, capthick=2)
plt.errorbar(30, 0.19, yerr=0.09, fmt='-o', color='orange', linestyle='None', capsize=5, capthick=2)
plt.errorbar(34, 0.16, yerr=0.07, fmt='-o', color='orange', linestyle='None', capsize=5, capthick=2)
plt.errorbar(39, 0.44, yerr=0.14, fmt='-o', color='blue', linestyle='None', capsize=5, capthick=2, label='Hadsall Creek - 21 years of infilling')

# Set title and labels
plt.title('Model Depth Validation')
plt.xlabel('$θ_H$')
plt.ylabel('Depth (m)')

# Adjust legend position
plt.legend(loc='upper left')

# Show plot
plt.show()

#%% Modeled depth over time for a specified hollow angle

# Depth ober time for a shallow (32) slope
# find your respective side slope value
# Knon was pulled from our modeled Knon function for respective hollow slope
slope = np.deg2rad(43)
hollow = np.arctan(0.8 * np.tan(slope))
t = np.arange(0, 3142) # This will have to be pulled from RI function as well
dep_shal = (
    (2 * 0.005966563315) ** 0.5
) * (np.cos(hollow) ** 0.5 * ((np.tan(slope) ** 2) - (np.tan(hollow) ** 2)) ** 0.25) * (
    ((np.cos(hollow) ** 2) / (np.cos(slope) ** 2) - 1) ** 0.25
) * (t ** 0.5)

# Depth ober time for a shallow (43) slope
# find your respective side slope value
# Knon was pulled from our modeled Knon function for respective hollow slope
slope_s = np.deg2rad(46.4)
hollow_s = np.arctan(0.8 * np.tan(slope_s))
t_s = np.arange(0, 372) # This will have to be pulled from RI function as well
dep_steep = (
    (2 * 0.013515216521) ** 0.5
) * (np.cos(hollow_s) ** 0.5 * ((np.tan(slope_s) ** 2) - (np.tan(hollow_s) ** 2)) ** 0.25) * (
    ((np.cos(hollow_s) ** 2) / (np.cos(slope_s) ** 2) - 1) ** 0.25
) * (t_s ** 0.5)

#%%
# Plot Depth over time for each scenrio for comparison
# Set the style
sns.set(style="whitegrid")

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)

# Plotting on the first subplot
sns.lineplot(x=t, y=dep_shal, ax=ax1, label="Shallow Depth")

# Plotting on the second subplot
sns.lineplot(x=t_s, y=dep_steep, ax=ax2, color='orange', label="Steep Depth")

# Set labels and title
ax1.set_xlabel("Time (yrs)")
ax1.set_ylabel("Depth (m)")
ax1.set_title("Depth over time")
ax1.legend()

ax2.set_xlabel("Time (yrs)")
ax2.set_ylabel("Depth (m)")
ax2.legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

