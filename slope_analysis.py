# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:57 2024

@author: sdavilao
"""

import pandas as pd

# Load data from an Excel file
file_path = 'your_file.xlsx'
sheet_name = 'Sheet1'  # Adjust the sheet name if necessary
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Group by 'ID' and calculate the mean of 'Value'
average_values = df.groupby('ID')['Value'].mean().reset_index()

# Save the averages into another DataFrame
df_averages = average_values.copy()

# Display the result
print(df_averages)

# Optionally, save the result to a new Excel file
output_file_path = 'averages_output.xlsx'
df_averages.to_excel(output_file_path, index=False)

