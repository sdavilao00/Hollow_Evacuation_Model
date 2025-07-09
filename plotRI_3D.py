# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:54:00 2025

@author: sdavilao
"""

import os
import pandas as pd
import glob
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Path to your CSV folder
folder_path = r"C:\Users\sdavilao\Documents\newcodesoil\results"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Regex to extract trailing number and "extX"
run_value_pattern = re.compile(r"(\d+)(?=\.csv$)", re.IGNORECASE)
extent_pattern = re.compile(r"(ext\d+)", re.IGNORECASE)

dfs = []

for file in csv_files:
    filename = os.path.basename(file)

    # Extract Run_Value from end of filename
    run_match = run_value_pattern.search(filename)
    # Extract extent (e.g., ext15)
    ext_match = extent_pattern.search(filename)

    if run_match and ext_match:
        run_value = int(run_match.group(1))
        extent = ext_match.group(1)
        df = pd.read_csv(file)
        df['Cohesion'] = run_value
        df['Extent'] = extent
        dfs.append(df)
    else:
        print(f"⚠️ Skipping file: {filename} (missing Run_Value or Extent)")

# Combine all into a single big DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Filter for slope > 28
filtered_df = combined_df[combined_df['Avg_Slope_deg'] > 25]

# Bin slopes by every 2 degrees again after filtering
bin_edges = np.arange(26, filtered_df['Avg_Slope_deg'].max() + 2, 2)
filtered_df['slope_bin'] = pd.cut(filtered_df['Avg_Slope_deg'], bins=bin_edges)
filtered_df['slope_mid'] = filtered_df['slope_bin'].apply(lambda x: x.mid)

filtered_df = filtered_df.dropna(subset=['slope_mid'])
filtered_df['slope_mid'] = filtered_df['slope_mid'].astype(float)

print("✅ Combined and filtered files with Extent and Run_Value.")

#%%

# Step 3: Line plot using binned slope midpoints
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=filtered_df,
    x='Avg_Slope_deg',
    y='Year',
    hue='Cohesion',
    palette='viridis',
    marker='o'
)

plt.xlabel("Slope Bin Midpoint (Degrees)")
plt.ylabel("Recurrence Interval (yrs)")
plt.yscale("log")
plt.title("Year vs. Binned Slope (2° bins), Colored by Run Value")
plt.grid(True)
plt.legend(title='Cohesion (Pa)')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(
    data=filtered_df,
    x='slope_mid',
    y='Avg_Soil_Depth_m',
    hue='Cohesion',
    palette='viridis',
    marker='o'
)

plt.xlabel("Slope Bin Midpoint (Degrees)")
plt.ylabel("Soil depth (m)")
#plt.yscale("log")
plt.title("Soil Depth vs. Binned Slope (2° bins), Colored by Run Value")
plt.grid(True)
plt.legend(title='Cohesion (Pa)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=filtered_df,
    x='Avg_Slope_deg',
    y='Optimal_Buffer_m',
    hue='Cohesion',
    palette='viridis',
    marker='o'
)

plt.xlabel("Slope Bin Midpoint (Degrees)")
plt.ylabel("Optimal Length (m)")
#plt.yscale("log")
plt.title("Year vs. Binned Slope (2° bins), Colored by Cohesion")
plt.grid(True)
plt.legend(title='Cohesion (Pa)')
plt.tight_layout()
plt.show()
#%% each cohesion seperately

# Step 2: Create FacetGrid to split by Run_Value
g = sns.FacetGrid(
    filtered_df,
    col='Cohesion',
    col_wrap=3,         # Adjust depending on number of Run_Values
    height=4,
    sharey=False        # Set to True if you want all plots on same y-scale
)

# Step 3: Add scatter plots
g.map_dataframe(sns.scatterplot, x='slope_mid', y='Year')

# Step 4: Adjust axes and add log scale to y-axis for all
for ax in g.axes.ravel():
    ax.set_yscale('log')
    ax.set_xlabel("Slope Bin Midpoint (Degrees)")
    ax.set_ylabel("Recurrence Interval (yrs)")
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

# Step 5: Title and layout
g.set_titles("Cohesion: {col_name}")
plt.suptitle("RI vs. Slope by Cohesion", y=1.05, fontsize=16)
plt.tight_layout()
plt.show()
#%% Regresson lines
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Ensure valid values
filtered_df = filtered_df.dropna(subset=['slope_mid', 'Year'])
filtered_df = filtered_df[filtered_df['Year'] > 0]
filtered_df['slope_mid'] = filtered_df['slope_mid'].astype(float)

# Store fit results
fit_results = []

for run_val, group in filtered_df.groupby('Cohesion'):
    x = group['slope_mid'].values
    y = group['Year'].values
    log_y = np.log(y)

    slope, intercept, r_value, p_value, std_err = linregress(x, log_y)
    a = np.exp(intercept)
    b = slope  # slope is negative for decay

    fit_results.append({
        'Run_Value': run_val,
        'a': round(a, 3),
        'b (decay rate)': round(-b, 5),
        'R_squared': round(r_value**2, 4)
    })

fit_df = pd.DataFrame(fit_results)
print(fit_df)

#%% Data and regress lines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup color palette
palette = sns.color_palette("viridis", n_colors=filtered_df['Cohesion'].nunique())
run_values = sorted(filtered_df['Cohesion'].unique())
color_map = dict(zip(run_values, palette))

plt.figure(figsize=(8,6))

# 1. Plot raw data
for run_val in run_values:
    group = filtered_df[filtered_df['Cohesion'] == run_val]
    plt.scatter(group['slope_mid'], group['Year'],
                color=color_map[run_val], label=f'{run_val}',
                alpha=0.5, edgecolor='k')

# 2. Overlay fitted exponential curves from fit_df
for i, row in fit_df.iterrows():
    run_val = row['Run_Value']
    a = row['a']
    b = row['b (decay rate)']  # already positive

    # Generate smooth x-values and compute fitted y
    x_fit = np.linspace(filtered_df['slope_mid'].min(), filtered_df['slope_mid'].max(), 100)
    y_fit = a * np.exp(-b * x_fit)

    plt.plot(x_fit, y_fit, color=color_map[run_val], linewidth=2)

# Final touches
#plt.yscale('log')
plt.xlabel("Slope Bin Midpoint (Degrees)")
plt.ylabel("Recurrence Interval (yrs)")
plt.title("RI vs. Slope with Exponential Fits by Cohesion")
plt.legend(title='Cohesion (Pa)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# # Choose a Run_Value group (or loop through multiple)
# group = filtered_df[filtered_df['Cohesion'] == 6400].copy()
# group = group[group['Year'] > 0]
# x = group['slope_mid'].astype(float)
# y = np.log(group['Year'])  # Fit in log space

# # Fit linear model
# X = sm.add_constant(x)
# model = sm.OLS(y, X).fit()

# # Predict and get CI in log space
# x_pred = np.linspace(x.min(), x.max(), 100)
# X_pred = sm.add_constant(x_pred)
# log_y_pred = model.predict(X_pred)

# # Get 95% confidence intervals
# predictions = model.get_prediction(X_pred)
# ci = predictions.conf_int(alpha=0.05)  # 95% CI

# # Convert back to original space
# y_pred = np.exp(log_y_pred)
# lower_ci = np.exp(ci[:, 0])
# upper_ci = np.exp(ci[:, 1])

# # Plot actual points and fitted exponential line with CI
# plt.figure(figsize=(8,6))
# plt.scatter(x, np.exp(y), alpha=0.5, edgecolor='k', label='Observed')
# plt.plot(x_pred, y_pred, label='Exponential Fit', color='blue')
# plt.fill_between(x_pred, lower_ci, upper_ci, color='blue', alpha=0.3, label='95% CI')

# plt.xlabel("Slope Bin Midpoint (Degrees)")
# plt.ylabel("Recurrence Interval (yrs)")
# plt.title("Exponential Fit with 95% CI (Year in Original Scale)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#%% ALL

# Prepare plot
plt.figure(figsize=(10, 6))
palette = plt.cm.viridis(np.linspace(0, 1, filtered_df['Cohesion'].nunique()))
run_vals = sorted(filtered_df['Cohesion'].unique())

# Loop over each Run_Value
for i, run_val in enumerate(run_vals):
    group = filtered_df[(filtered_df['Cohesion'] == run_val) & (filtered_df['Year'] > 0)].copy()
    
    if len(group) < 3:
        continue  # skip if too few points
    
    x = group['slope_mid'].astype(float)
    y_log = np.log(group['Year'])

    # Fit in log space
    X = sm.add_constant(x)
    model = sm.OLS(y_log, X).fit()
    
    # Predict and get CI
    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_log_pred = model.predict(X_pred)
    
    ci = model.get_prediction(X_pred).conf_int()
    
    # Back-transform to original scale
    y_pred = np.exp(y_log_pred)
    lower = np.exp(ci[:, 0])
    upper = np.exp(ci[:, 1])
    
    color = palette[i]
    
    # Plot points
    plt.scatter(x, np.exp(y_log), alpha=0.5, edgecolor='k', label=f'{run_val}', color=color)
    
    # Plot fit and CI band
    plt.plot(x_pred, y_pred, color=color, linewidth=2)
    plt.fill_between(x_pred, lower, upper, color=color, alpha=0.2)

# Final plot formatting
plt.xlabel("Slope Bin Midpoint (Degrees)")
plt.ylabel("Recurrence Interval (yrs)")
plt.title("Exponential Regression with 95% CI")
plt.yscale("log")  # Optional, depending on spread
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend(title='Cohesion (Pa)')
plt.tight_layout()
plt.show()


#%% individual with CI
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Unique Run_Values
run_vals = sorted(filtered_df['Cohesion'].unique())
n = len(run_vals)

# Create side-by-side subplots
fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(5 * n, 5), sharey=True)

# Handle case where n = 1 (axes not iterable)
if n == 1:
    axes = [axes]

for i, run_val in enumerate(run_vals):
    ax = axes[i]
    group = filtered_df[(filtered_df['Cohesion'] == run_val) & (filtered_df['Year'] > 0)].copy()

    if len(group) < 3:
        ax.set_title(f"Run {run_val} (insufficient data)")
        ax.axis('off')
        continue

    x = group['slope_mid'].astype(float)
    y_log = np.log(group['Year'])

    # Fit exponential decay in log space
    X = sm.add_constant(x)
    model = sm.OLS(y_log, X).fit()

    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    y_log_pred = model.predict(X_pred)
    ci = model.get_prediction(X_pred).conf_int()

    y_pred = np.exp(y_log_pred)
    lower = np.exp(ci[:, 0])
    upper = np.exp(ci[:, 1])

    # Plot
    ax.scatter(x, np.exp(y_log), alpha=0.5, edgecolor='k', label='Data')
    ax.plot(x_pred, y_pred, color='blue', label='Exp Fit')
    ax.fill_between(x_pred, lower, upper, color='blue', alpha=0.2, label='95% CI')

    ax.set_title(f"Cohesion (Pa): {run_val}")
    ax.set_xlabel("Slope Bin Midpoint (Degrees)")
    ax.set_yscale("log")
    ax.set_ylim(1e0, 1e5)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    if i == 0:
        ax.set_ylabel("Recurrence Interval (yrs)")
    else:
        ax.set_ylabel("")

    ax.legend()

# Adjust layout and title
fig.suptitle("Exponential Fit with 95% CI", fontsize=16, y=1.05)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the suptitle
plt.show()


