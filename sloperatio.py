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

# read in hollow axis excel file
path = "C:\\Users\\12092\\Documents\\Hollow_Data\\hollow_slopes\\HadsallAvg9AxisStats_0614.xlsx"

"HOLLOW ANGLE"
axis_df = pd.read_excel(path, names=['LINEID', 'ID', 'DIST', 'DIST_SURF', 'X', 'Y', 'ANGLE'])
#print(axis_df)

# get the mean of every polyline for the hollow axis
key = axis_df['LINEID']
axis_mean = axis_df.groupby(key)['ANGLE'].mean()
axis_median = axis_df.groupby(key)['ANGLE'].median()


# read in side slope  excel file
path = "C:\\Users\\12092\\Documents\\Hollow_Data\\hollow_slopes\\HadsallAvg9SideslopeStats_0614.xlsx"

"SIDE SLOPE ANGLE"
side_df = pd.read_excel(path, names=['LINEID', 'ID', 'DIST', 'DIST_SURF', 'X', 'Y', 'ANGLE'])
#print(side_df)

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

## Fit a regression line
coefficients = np.polyfit(np.tan(np.deg2rad(comp_df.sideslope_mean)), np.tan(np.deg2rad(comp_df.hollow_axis_mean)), 1)
poly = np.poly1d(coefficients)

# Generate x values for the regression line
x_regression = np.linspace(min(np.tan(np.deg2rad(comp_df.sideslope_mean))), max(np.tan(np.deg2rad(comp_df.sideslope_mean))), 100)


# Plot the regression line
plt.plot(x_regression, poly(x_regression), label='Regression Line', color='red')

## Plot this up
## y is tan of hollow slope, x is tan of side slope

plt.figure()
plt.title('Slope Ratio')
plt.scatter((np.tan(np.deg2rad(comp_df.sideslope_mean))), (np.tan(np.deg2rad(comp_df.hollow_axis_mean))))
plt.ylabel('tan($θ_H$)')
plt.xlabel('tan($θ_S$)')
# Plot the regression line
plt.plot(x_regression, poly(x_regression), label='Regression Line', color='red')
equation = f"y = {round(coefficients[0], 2)}x + {round(coefficients[1], 2)}"
plt.text(0.3, 0.7, equation, color='red', fontsize=12)

#comp_df.to_excel('C:\\Users\\12092\\Documents\\Hallow_Evacuation_Data\\comp_df.xlsx', index=False)
##############################################################################

# ratio = (np.tan(np.deg2rad(comp_df.hollow_axis_mean)))/(np.tan(np.deg2rad(comp_df.sideslope_mean)))

# plt.figure()
# plt.title('Slope Ratio')
# plt.scatter(comp_df.hollow_axis_mean, ratio)
# plt.xlabel('$θ_S$')
# plt.ylabel('tan($θ_H$)/tan($θ_S$)')
