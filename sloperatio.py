# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:32:08 2023

@author: Selina Davila Olivera

Code for streamlining statistical analysis on hollow axis and side slopes

From there it will create plots to determine a relationship (if any) between the two
"""
## Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"HOLLOW ANGLE"
# read in hollow axis excel file
path = "C:\\Users\\12092\\Documents\\QGIS_Hollow_Data\\HadsallAvg9AxisStats.xlsx"

axis_df = pd.read_excel(path, names=['LINEID', 'ID', 'DIST', 'DIST_SURF', 'X', 'Y', 'ANGLE'])
print(axis_df)

# get the mean of every polyline for the hollow axis
key = axis_df['LINEID']
axis_mean = axis_df.groupby(key)['ANGLE'].mean()
axis_median = axis_df.groupby(key)['ANGLE'].median()

"SIDE SLOPE ANGLE"
# read in side slope  excel file
path = "C:\\Users\\12092\\Documents\\QGIS_Hollow_Data\\HadsallAvg9SideslopeStats.xlsx"

side_df = pd.read_excel(path, names=['LINEID', 'ID', 'DIST', 'DIST_SURF', 'X', 'Y', 'ANGLE'])
print(side_df)

# get the mean of every polyline for the side slopes
key = side_df['LINEID']
x = side_df.groupby(key)['ANGLE'].mean()
y = side_df.groupby(key)['ANGLE'].median()

# every 2 LINEID's refers to one hollow, so avergae the two side slope angles
side_mean = x.groupby(np.arange(len(x))//2).mean()
side_median = y.groupby(np.arange(len(x))//2).mean()

## Create DataFrame
comp_df = pd.DataFrame(zip(axis_mean, axis_median, side_mean, side_median))
## rename columns
comp_df.columns = ['hollow_axis_mean', 'hollow_axis_median', 'sideslope_mean', 'sideslope_median']

## Plot this up
## y is tan of hollow slope, x is tan of side slope

plt.figure()
plt.scatter((np.tan(np.deg2rad(comp_df.sideslope_mean))), (np.tan(np.deg2rad(comp_df.hollow_axis_mean))))
plt.ylabel('tan(hollow angle)')
plt.xlabel('tan(side slope angle)')

##############################################################################

ratio = (np.tan(np.deg2rad(comp_df.hollow_axis_mean)))/(np.tan(np.deg2rad(comp_df.sideslope_mean)))

plt.figure()
plt.scatter(comp_df.sideslope_mean, ratio)
plt.xlabel('Side slope angle (deg)')
plt.ylabel('tan(hollow ang)/tan(side slope ang)')
