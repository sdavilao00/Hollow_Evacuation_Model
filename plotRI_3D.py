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
folder_path = r"C:\Users\sdavilao\Documents\newcodesoil\results\new"
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

#remove bad hollows
#filtered_df = filtered_df.drop([82,84,86,88,110,111,112,113])
filtered_df = filtered_df.drop([38,40,44])

# Bin slopes by every 2 degrees again after filtering
bin_edges = np.arange(26, filtered_df['Avg_Slope_deg'].max() + 2, 2)
filtered_df['slope_bin'] = pd.cut(filtered_df['Avg_Slope_deg'], bins=bin_edges)
filtered_df['slope_mid'] = filtered_df['slope_bin'].apply(lambda x: x.mid)

filtered_df = filtered_df.dropna(subset=['slope_mid'])
filtered_df['slope_mid'] = filtered_df['slope_mid'].astype(float)

print("✅ Combined and filtered files with Extent and Run_Value.")

#%% SATURATION
# ----------------------------
# second folder with fixed m and Cohesion (now parsed from filename)
# ----------------------------
m_folder_path = r"C:\Users\sdavilao\Documents\newcodesoil\results\new\m"
m_csv_files = glob.glob(os.path.join(m_folder_path, "*.csv"))
m_pattern = re.compile(r"ext(\d+)_([0-9]+)_mp([0-9]+)", re.IGNORECASE)

m_dfs = []
for file in m_csv_files:
    filename = os.path.basename(file)
    m_match = m_pattern.search(filename)

    if m_match:
        extent = int(m_match.group(1))          # X from extX
        cohesion = int(m_match.group(2))        # 1920 or 760
        m_val = int(m_match.group(3)) / 100.0   # mp85 -> 0.85

        df_m = pd.read_csv(file)
        df_m['Extent'] = extent
        df_m['Cohesion'] = cohesion
        df_m['m'] = m_val

        m_dfs.append(df_m)
    else:
        print(f"⚠️ Skipping m-folder file: {filename} (pattern not matched)")

# Create m_df (could be empty if none matched)
m_df = pd.concat(m_dfs, ignore_index=True) if m_dfs else pd.DataFrame()
m_df = m_df.drop([24,31, 33])
# ----------------------------
# Downstream filtering (unchanged, except now uses new m_df)
# ----------------------------
# Filter for slope > 25
filtered_df = combined_df[combined_df['Avg_Slope_deg'] > 25]
filtered_df = filtered_df.dropna(subset=['slope_mid'])
filtered_df['slope_mid'] = filtered_df['slope_mid'].astype(float)

print("Combined and filtered files with Extent, Cohesion, and m parsed from filename.")


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

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data
df = filtered_df

# Log-space inverse model
def inverse_model_log(theta, loga, b):
    return loga - np.log10(theta + b)

# Color-blind-friendly palette
cbf_colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7', '#0072B2']

plt.figure(figsize=(9, 6))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")
ax.grid(True, alpha=0.4)

legend_entries = []

cohesions = sorted(df["Cohesion"].unique(), reverse=True)

for i, coh in enumerate(cohesions):
    g = df[df["Cohesion"] == coh]
    x = g["Avg_Slope_deg"].values
    y = g["Year"].values
    color = cbf_colors[i % len(cbf_colors)]

    # Scatter
    ax.scatter(x, y, color=color, s=60, alpha=0.75)

    if len(x) < 3:
        continue

    # Fit in log space
    popt, _ = curve_fit(
        inverse_model_log,
        x,
        np.log10(y),
        p0=[3.0, -25]
    )

    loga_fit, b_fit = popt
    a_fit = 10**loga_fit

    # Log-space R²
    y_log = np.log10(y)
    y_log_pred = inverse_model_log(x, loga_fit, b_fit)
    ss_res = np.sum((y_log - y_log_pred) ** 2)
    ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
    r2_log = 1 - ss_res / ss_tot

    # Smooth curve
    x_fit = np.linspace(min(x) - 0.5, max(x) + 0.5, 300)
    y_fit = a_fit / (x_fit + b_fit)

    line, = ax.plot(x_fit, y_fit, color=color, lw=2.5)

    label = (
        f"{int(coh)} Pa: "
        f"$RI=\\frac{{{a_fit:.1f}}}{{\\theta_H+{b_fit:.2f}}}$, "
        f"$R^2_{{\\log}}={r2_log:.2f}$"
    )
    legend_entries.append((line, label))

# Axes
ax.set_yscale("log")
ax.set_xlabel("Hollow slope, $\\theta_H$ (°)", fontsize=13, fontweight="bold")
ax.set_ylabel("Recurrence interval, RI (years)", fontsize=13, fontweight="bold")

# Legend
if legend_entries:
    handles, labels = zip(*legend_entries)
    ax.legend(handles, labels, title="Cohesion (log-space fit)",
              fontsize=10, title_fontsize=11,
              loc="upper right", frameon=True)

# X ticks
xticks = np.arange(
    np.floor(df["Avg_Slope_deg"].min() / 3) * 3,
    df["Avg_Slope_deg"].max() + 1, 3
)
ax.set_xticks(xticks)

plt.tight_layout()
#plt.savefig("C:/Users/sdavilao/Documents/RI_3D_Fit.png", dpi=450, bbox_inches='tight')
plt.show()


#%% RI vs Hollow slope FINAL
# Re-import necessary libraries after code execution environment reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define inverse model
def inverse_model(x, a, b):
    return a / (x + b)

# Use a color-blind-friendly palette (CUD palette)
cbf_colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00']  # Orange, Sky Blue, Bluish Green, Vermillion

# Replot using the updated color palette
plt.figure(figsize=(10, 6))
plt.gca().set_facecolor("#f0f0f0")

plt.grid(True)

legend_entries = []

# Sort groups by descending cohesion
cohesion_groups = sorted(filtered_df.groupby("Cohesion"), key=lambda x: x[0], reverse=True)

for i, (cohesion, group) in enumerate(cohesion_groups):
    x = group["Avg_Slope_deg"].values
    y = group["Year"].values
    color = cbf_colors[i % len(cbf_colors)]

    # Fit inverse model
    popt, _ = curve_fit(inverse_model, x, y, p0=[1000, -25])
    a_fit, b_fit = popt

    # Compute R²
    residuals = y - inverse_model(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # LaTeX equation string
    equation_str = (
        f"${int(cohesion)}\\,\\mathrm{{Pa}}:\\quad RI = \\frac{{{a_fit:.1f}}}{{\\theta_H + {b_fit:.2f}}}"
        f"\\quad R^2 = {r_squared:.3f}$"
    )

    # Plot data (change marker size here)
    plt.scatter(x, y, alpha=0.8, color=color, s=60)  # <--- marker size

    # Plot fitted curve (change line width here)
    x_fit = np.linspace(27, max(x) + 1, 200)
    y_fit = inverse_model(x_fit, a_fit, b_fit)
    line, = plt.plot(
        x_fit, y_fit,
        color=color,
        linestyle='--',
        linewidth=2.5,       # <--- line width
        label=equation_str
    )

    legend_entries.append((line, equation_str))


# Set axis properties
plt.yscale("log")
# plt.xlabel("Hollow Slope, $\\theta_H$ (°)", fontsize=14, fontweight='bold')
# plt.ylabel("Recurrence Interval, RI (years)", fontsize=14, fontweight='bold')

# Add legend in descending cohesion order
# handles, labels = zip(*legend_entries)
# legend = plt.legend(handles, labels, title="Cohesion Fit Equation", fontsize=14, title_fontsize=15,
#             loc='upper right', bbox_to_anchor=(0.98, 1.01), frameon=True)
# legend.get_frame().set_linewidth(3)

tick_range = np.arange(27, filtered_df["Avg_Slope_deg"].max() + 1, 3)
plt.xticks(tick_range)

# Make all spines (box edges) thicker
# for spine in plt.gca().spines.values():
#     spine.set_linewidth(2)  # increase thickness as needed
    
#plt.tick_params(axis='both', labelsize=12, width=2)
# for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#     label.set_fontweight('bold')

plt.tight_layout()
#plt.savefig("C:/Users/sdavilao/Documents/RI_3D.png", dpi=450, bbox_inches='tight')
plt.show()


#%% SOIL DEPTH
# Replot using dot markers instead of x's
plt.figure(figsize=(10, 6))
plt.gca().set_facecolor("#f0f0f0")

# Reuse the color-blind-friendly palette and sort by descending cohesion
for i, (cohesion, group) in enumerate(cohesion_groups):
    x = group["Avg_Slope_deg"].values
    y = group["Avg_Soil_Depth_m"].values
    color = cbf_colors[i % len(cbf_colors)]

    # Use circle ('o') marker
    plt.scatter(x, y, alpha=0.7, color=color, marker='o', label=f"{int(cohesion)} Pa")

# Axes styling
#plt.xlabel("Average Slope $\\theta_H$ (degrees)", fontsize=14, fontweight='bold')
#plt.ylabel("Average Soil Depth (m)", fontsize=14, fontweight='bold')
# tick_range = np.arange(27, filtered_df["Avg_Slope_deg"].max() + 1, 2)
# plt.xticks(tick_range)
plt.tick_params(axis='both')
# for spine in plt.gca().spines.values():
#     spine.set_linewidth(2)
# for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
#     label.set_fontweight('bold')

# Legend
# legend = plt.legend(title="Cohesion", fontsize=14, title_fontsize=15,
#                     loc='upper right', bbox_to_anchor=(0.98, 1.01), frameon=True)
# legend.get_frame().set_linewidth(2)

plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/sdavilao/Documents/3D_soildepth.png", dpi=450, bbox_inches='tight')
plt.show()

#%% Combined saturation and cohesion plot 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Define inverse model
def inverse_model(x, a, b):
    return a / (x + b)

# Refit the inverse model and plot combined results
inverse_fit_results = []

plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red']

for i, (cohesion, group) in enumerate(filtered_df.groupby("Cohesion")):
    x = group["Avg_Slope_deg"].values
    y = group["Year"].values
    color = colors[i % len(colors)]

    # Fit the inverse model
    popt, _ = curve_fit(inverse_model, x, y, p0=[1000, -25])
    a_fit, b_fit = popt

    # Store results
    inverse_fit_results.append({
        "Cohesion": cohesion,
        "a": a_fit,
        "b": b_fit
    })

    # Plot original data
    plt.scatter(x, y, label=f"{cohesion} Pa (data)", alpha=0.5, color=color)

    # Plot fitted curve only from slope = 27.5 and above
    x_fit = np.linspace(27.5, max(x) + 1, 200)
    y_fit = inverse_model(x_fit, a_fit, b_fit)
    plt.plot(x_fit, y_fit, color=color, linestyle='-', label=f"{cohesion} Pa (fit)")

# Overlay m_df data
# Overlay m_df data
if not m_df.empty:
    m_plot = m_df[m_df["Cohesion"] == 760]   # filter here

    sns.scatterplot(
        data=m_plot,              # <-- use m_plot, not m_df
        x="Avg_Slope_deg",
        y="Year",
        style="m",
        s=60,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
        color="black",            # optional
        label="m-data (C=760)"
    )


# Finalize combined plot
plt.yscale("log")
plt.xlabel("Average Slope (degrees)")
plt.ylabel("Recurrence Interval (years, log scale)")
plt.title("Inverse Model Fit and Extent Data (S ≥ 27.5°)")
plt.legend(title="Cohesion / Saturation", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Display fit parameters
fit_results_df = pd.DataFrame(inverse_fit_results)
fit_results_df

#%%
# Re-import necessary libraries after code execution environment reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------- Inverse models --------
def inverse_model(x, a, b):
    """Unbounded inverse model: RI = a / (x + b)"""
    return a / (x + b)

def inverse_model_fixed(x, a, theta_crit):
    """Bounded inverse model: RI = a / (x - theta_crit)"""
    return a / (x - theta_crit)

# Use a color-blind-friendly palette (CUD palette)
cbf_colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00']  # Orange, Sky Blue, Bluish Green, Vermillion

# Critical slope (deg) for each cohesion you want bounded
crit_slopes = {
    760: 28.0,
    1920: 29.0,
    # add more if needed, e.g. 6400: 27.0
}

# --- Combine the two dataframes: m_df (m=0.85) and filtered_df (m=1.0) ---
# Assuming both already have an 'm' column; if not, you can explicitly assign:
# m_df = m_df.assign(m=0.85)
# filtered_df = filtered_df.assign(m=1.0)

combined_df = pd.concat([m_df, filtered_df], ignore_index=True)

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")
plt.grid(True)

legend_entries = []

# Unique cohesions and saturations
unique_coh = sorted(combined_df["Cohesion"].unique(), reverse=True)
unique_m = sorted(combined_df["m"].unique())

# Map cohesion to color (CUD palette)
color_map = {coh: cbf_colors[i % len(cbf_colors)] for i, coh in enumerate(unique_coh)}

# Map saturation to linestyle / marker
linestyle_map = {
    0.85: '--',
    1.00: '-',
}
marker_map = {
    0.85: 'o',
    1.00: 's',
}

# Group by (Cohesion, m)
grouped = combined_df.groupby(["Cohesion", "m"])

for (cohesion, m_val), group in sorted(grouped, key=lambda x: (x[0][0], x[0][1]), reverse=True):
    x = group["Avg_Slope_deg"].values
    y = group["Year"].values

    color = color_map[cohesion]
    linestyle = linestyle_map.get(float(m_val), '-.')
    marker = marker_map.get(float(m_val), 'o')

    # Scatter points
    plt.scatter(
        x, y,
        alpha=0.8,
        color=color,
        s=60,
        marker=marker,
    )

    theta_crit = crit_slopes.get(cohesion, None)

    if theta_crit is not None:
        # --- BOUNDED CASE: RI = a / (θ - θ_crit) ---
        fit_mask = x > (theta_crit + 0.1)
        xdata = x[fit_mask]
        ydata = y[fit_mask]

        if len(xdata) > 2:
            def model_for_fit(x_local, a):
                return inverse_model_fixed(x_local, a, theta_crit)

            popt, _ = curve_fit(model_for_fit, xdata, ydata, p0=[1000])
            a_fit = popt[0]

            # R² for this (C, m) combo using the fit data
            y_pred = model_for_fit(xdata, a_fit)
            residuals = ydata - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata - np.mean(ydata))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Smooth curve for plotting
            x_fit = np.linspace(theta_crit + 0.1, max(x) + 1, 300)
            y_fit = model_for_fit(x_fit, a_fit)

            line, = plt.plot(
                x_fit, y_fit,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
            )

            equation_str = (
                f"${int(cohesion)}\\,\\mathrm{{Pa}},\\ m={m_val:.2f}:\\ "
                f"RI = \\frac{{{a_fit:.1f}}}{{\\theta_H - {theta_crit:.1f}}}"
                f"\\quad R^2 = {r_squared:.3f}$"
            )

            legend_entries.append((line, equation_str))
        else:
            print(f"⚠️ Not enough points above θ_crit={theta_crit:.1f}° to fit C={cohesion} Pa, m={m_val:.2f}")

    else:
        # --- UNBOUNDED FALLBACK: RI = a / (θ + b) ---
        if len(x) > 2:
            popt, _ = curve_fit(inverse_model, x, y, p0=[1000, -25])
            a_fit, b_fit = popt

            y_pred = inverse_model(x, *popt)
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)

            x_fit = np.linspace(27, max(x) + 1, 300)
            y_fit = inverse_model(x_fit, a_fit, b_fit)

            line, = plt.plot(
                x_fit, y_fit,
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
            )

            equation_str = (
                f"${int(cohesion)}\\,\\mathrm{{Pa}},\\ m={m_val:.2f}:\\ "
                f"RI = \\frac{{{a_fit:.1f}}}{{\\theta_H + {b_fit:.2f}}}"
                f"\\quad R^2 = {r_squared:.3f}$"
            )

            legend_entries.append((line, equation_str))
        else:
            print(f"⚠️ Not enough points to fit C={cohesion} Pa, m={m_val:.2f} with unbounded model")

# ---- Axis & legend formatting ----
plt.yscale("log")
plt.xlabel("Hollow Slope, $\\theta_H$ (°)", fontsize=14, fontweight='bold')
plt.ylabel("Recurrence Interval, RI (years)", fontsize=14, fontweight='bold')

if legend_entries:
    handles, labels = zip(*legend_entries)
    legend = plt.legend(
        handles, labels,
        title="Cohesion & m Fit Equation",
        fontsize=12,
        title_fontsize=13,
        loc='upper right',
        bbox_to_anchor=(0.98, 1.01),
        frameon=True
    )
    legend.get_frame().set_linewidth(3)

tick_range = np.arange(27, combined_df["Avg_Slope_deg"].max() + 1, 3)
plt.xticks(tick_range)

plt.tight_layout()
plt.show()

#%% CRitical area vs volume vs erosion

df_1920 = filtered_df[ filtered_df["Cohesion"] == 1920 ]

# Critical Depth variables
K = 0.0042
Sc = 1.25
pw = 1000 # kg/m3 # density of water
ps = 1600 # kg/m3c # density of soil
g = 9.81  # m/s2 # force of gravity
yw = g*pw
ys = g*ps
phi = np.deg2rad(41)  # converts phi to radians ## set to 42 right now
# z = np.arange(0,6,0.1)
z = df_1920['Avg_Soil_Depth_m']

# Slope stability variables
m = 1 # m # saturation ration (h/z)
l = 10 # m # length
w = 6.7 # m # width
C0 = 1920 # Pa
#C0 = filtered_df['Cohesion']
j = 0.8

#Define side/hollow slope range
# slope_ang = np.arange(27,51, 0.1) # Side slope range from 27 to 51 in degrees in 0.1 intervals
# slope_rad = np.deg2rad(slope_ang) # Side slope in radians
# hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang)))))) # Hollow slope in radians
# hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang)))))) # Hollow slope in degrees
hollow_rad = np.deg2rad(df_1920['Avg_Slope_deg'])
slope_rad = np.deg2rad((df_1920['Avg_Slope_deg'])/0.89)


#Cohesion variables
Crb = C0*2.718281**(-z*j)
Crl = (C0/(j*z))*(1 - 2.718281**(-z*j))

K0 = 1 - np.sin(hollow_rad)

Kp = np.tan((np.deg2rad(45))+(phi/2))**2
Ka = np.tan((np.deg2rad(45))-(phi/2))**2
#%% Area

# Define terms of equation
A = (2*Crl*z + K0*(z**2)*(ys-yw*(m**2))*np.tan(phi))*np.cos(hollow_rad)*(l/w)**0.5
B = (Kp-Ka)*0.5*(z**2)*(ys-yw*(m**2))*(l/w)**(-0.5)
C = (np.sin(hollow_rad)*np.cos(hollow_rad)*z*ys) - Crb - (((np.cos(hollow_rad))**2)*z*(ys-yw*m)*np.tan(phi))

#Find critical area
Ac = ((A + B)/C)**2

Ac_2 = ((df_1920['Optimal_Buffer_m']) *3.14)**2
vol_2 = df_1920['Avg_Soil_Depth_m'] * ((df_1920['Optimal_Buffer_m']) *3.14)**2

df_1920["Ac"] = Ac
df_1920['Ac2'] = Ac_2
df_1920["Volume"] = Ac *z
df_1920['vol2'] = vol_2

#%%
cmap = plt.cm.viridis  

groups = filtered_df.groupby("Cohesion")
colors = cmap(np.linspace(0, 1, len(groups)))

for color, (label, sub) in zip(colors, groups):
    plt.scatter(sub["qs"], sub["Avg_Soil_Depth_m" * 'Optimal_Buffer_m'],
             label=label,
             color=color,
             marker="o")

plt.legend(title="Category")
plt.xlabel("Erosion")
plt.ylabel("Volume")
#plt.title("Value over time by Category")
plt.show()

#%%
cmap = plt.cm.viridis  

groups = filtered_df.groupby("Cohesion")
colors = cmap(np.linspace(0, 1, len(groups)))

# for color, (label, sub) in zip(colors, groups):
    
#     # scale marker sizes by Volume
#     sizes = sub["Volume"]   # raw values
#     # (optional) apply scaling so they don't get too big:
#     sizes = sizes * 1       # tweak this number if needed
    
plt.scatter(
        sub["qs"], sub["Volume"],
        label=label,
        color=color,
        #s=sizes,          # << size mapped to Volume
        alpha=0.8,
        edgecolors="k",
)

plt.legend(
    title="Volume Cohesion",
    loc="best",
    fontsize=10,
    title_fontsize=11,
    labelspacing=1.2,   # vertical space between entries
    handlelength=2.5,   # length of legend markers/lines
    borderpad=1.2       # padding inside the box
)
plt.yscale('log')
plt.xlabel("Erosion Rate")
plt.ylabel("Volume")
#plt.title("Volume vs qs by Cohesion")
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

cmap = plt.cm.viridis

groups = filtered_df.groupby("Cohesion")
colors = cmap(np.linspace(0, 1, len(groups)))

size_scale = 1

fig, ax = plt.subplots()

# ---- Plot + cohesion proxy handles ----
cohesion_handles = []
for color, (label, sub) in zip(colors, groups):
    ax.scatter(
        sub["qs"], sub["Year"],
        color=color,
        s=sub["Volume"] * size_scale,
        alpha=0.8,
        edgecolors="k"
    )
    cohesion_handles.append(
        Line2D([], [], marker="o", linestyle="None",
               color=color, markersize=8, label=str(label))
    )

# ---- Volume size handles: min / mid / max (actual values) ----
vol_unique = np.sort(filtered_df["Volume"].dropna().unique())
v_min = vol_unique[0]
v_mid = vol_unique[len(vol_unique)//2]
v_max = vol_unique[-1]

volume_handles = [
    Line2D([], [], marker="o", linestyle="None", color="gray",
           markersize=np.sqrt(v_min * size_scale), label=f"{round(v_min)}"),
    Line2D([], [], marker="o", linestyle="None", color="gray",
           markersize=np.sqrt(v_mid * size_scale), label=f"{round(v_mid)}"),
    Line2D([], [], marker="o", linestyle="None", color="gray",
           markersize=np.sqrt(v_max * size_scale), label=f"{round(v_max)}")
]

# ---- Column headers ----
coh_header = Line2D([], [], linestyle="None", label="Cohesion")
vol_header = Line2D([], [], linestyle="None", label="Volume")



# ---- Pad columns so they align vertically ----
n_rows = max(len(cohesion_handles), len(volume_handles))

def pad(handles, n):
    return handles + [Line2D([], [], linestyle="None", label="")] * (n - len(handles))

cohesion_col = [coh_header] + pad(cohesion_handles, n_rows)
volume_col   = [vol_header] + pad(volume_handles, n_rows)

# ---- IMPORTANT: concatenate column-wise (not interleaved) ----
legend_handles = cohesion_col + volume_col

ax.legend(
    handles=legend_handles,
    ncol=2,
    loc="upper right",
    frameon=True,
    columnspacing=2.0,
    handletextpad=0.8,
    labelspacing=1.1,
    borderpad=1.2
)

ax.set_yscale('log')
ax.set_xlabel("Erosion")
ax.set_ylabel("Recurrence Interval")
#ax.set_title("Volume vs qs by Cohesion")

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

cmap = plt.cm.viridis

groups = filtered_df.groupby("Cohesion")
colors = cmap(np.linspace(0, 1, len(groups)))

size_scale = 1

fig, ax = plt.subplots()

# ---- Plot + cohesion proxy handles ----
cohesion_handles = []
for color, (label, sub) in zip(colors, groups):
    ax.scatter(
        sub["Avg_Slope_deg"], sub["Year"],
        color=color,
        s=sub["Volume"] * size_scale,
        alpha=0.8,
        edgecolors="k"
    )
    cohesion_handles.append(
        Line2D([], [], marker="o", linestyle="None",
               color=color, markersize=8, label=str(label))
    )

# ---- Volume size handles: min / mid / max (actual values) ----
vol_unique = np.sort(filtered_df["Volume"].dropna().unique())
v_min = vol_unique[0]
v_mid = vol_unique[len(vol_unique)//2]
v_max = vol_unique[-1]

volume_handles = [
    Line2D([], [], marker="o", linestyle="None", color="gray",
           markersize=np.sqrt(v_min * size_scale), label=f"{round(v_min)}"),
    Line2D([], [], marker="o", linestyle="None", color="gray",
           markersize=np.sqrt(v_mid * size_scale), label=f"{round(v_mid)}"),
    Line2D([], [], marker="o", linestyle="None", color="gray",
           markersize=np.sqrt(v_max * size_scale), label=f"{round(v_max)}")
]

# ---- Column headers ----
coh_header = Line2D([], [], linestyle="None", label="Cohesion")
vol_header = Line2D([], [], linestyle="None", label="Volume")



# ---- Pad columns so they align vertically ----
n_rows = max(len(cohesion_handles), len(volume_handles))

def pad(handles, n):
    return handles + [Line2D([], [], linestyle="None", label="")] * (n - len(handles))

cohesion_col = [coh_header] + pad(cohesion_handles, n_rows)
volume_col   = [vol_header] + pad(volume_handles, n_rows)

# ---- IMPORTANT: concatenate column-wise (not interleaved) ----
legend_handles = cohesion_col + volume_col

ax.legend(
    handles=legend_handles,
    ncol=2,
    loc="upper right",
    frameon=True,
    columnspacing=2.0,
    handletextpad=0.8,
    labelspacing=1.1,
    borderpad=1.2
)

ax.set_yscale('log')
ax.set_xlabel("Hollow Slope")
ax.set_ylabel("Recurrence Interval")
#ax.set_title("Volume vs qs by Cohesion")

plt.show()

