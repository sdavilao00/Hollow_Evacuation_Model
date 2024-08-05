# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:57 2024

@author: sdavilao
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Correct file path
file_path = 'C:/Users/sdavilao/Documents/slopecomp.xlsx'
sheet_name = 'wb_2000'

# Read the Excel file
df = pd.read_excel(file_path, sheet_name = sheet_name)

    
# Group by 'ID' and calculate the mean of 'Value'
average_values_basin = df.groupby('ID')['_mean'].mean().reset_index()
average_values_sideslope = df.groupby('ID')['_average'].mean().reset_index()
    
# Save the averages into another DataFrame
df1 = average_values_basin.copy()
df_averages = pd.merge(df1, average_values_sideslope, on='ID', how='left')
    

    
    # # Optionally, save the result to a new Excel file
    # output_file_path = 'C:/Users/sdavilao/Documents/averages_output.xlsx'
    # df_averages.to_excel(output_file_path, index=False)
    # print(f"Averages saved to {output_file_path}")
    

plt.figure()
plt.scatter(df._mean, df._average)

sns.regplot(x=df._mean, y=df._average, data=df, ci=95, color='red', scatter=False, label='Regression Line')
# Calculate the regression line equation
slope, intercept, r_value, p_value, std_err = stats.linregress(df['_mean'], df['_average'])
equation = f"Y = {slope:.2f}X + {intercept:.2f}"
# Annotate the equation inside the plot
# Dynamically calculate position
x_pos = min(df['_mean']) + (max(df['_mean']) - min(df['_mean'])) * 0.05
y_pos = min(df['_average']) + (max(df['_average']) - min(df['_average'])) * 0.9
# Annotate the equation on the plot
plt.text(x_pos, y_pos, equation, fontsize=12, color='red')
# Print the equation of the regression line
print(f"Regression line equation: Y = {slope:.2f}X + {intercept:.2f}")
plt.xlabel('Basin Average Slope')
plt.ylabel("Average Side Slope")