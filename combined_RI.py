# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:48:19 2025

@author: sdavilao
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Option 1 — File in same folder:
# combined_df = pd.read_excel("combined_df.xlsx")

# Option 2 — Full path (edit as needed):
combined_df = pd.read_excel(r"C:\Users\sdavilao\Documents\combined_df.xlsx")

# Check
print("Loaded rows:", len(combined_df))
print(combined_df.head())

# ------------ Inverse model helpers ------------

def inverse_model(x, a, b):
    """
    Unbounded inverse model: RI = a / (x + b)
    """
    return a / (x + b)

def inverse_model_fixed(x, a, theta_crit):
    """
    Bounded inverse model: RI = a / (x - theta_crit)
    with vertical asymptote at theta_crit (deg).
    """
    return a / (x - theta_crit)


def plot_ri_vs_slope_by_cohesion_and_saturation(combined_df):
    """
    combined_df must have columns:
      - 'Cohesion' (Pa)
      - 'm' (relative saturation)
      - 'Avg_Slope_deg' (deg)
      - 'Year' (RI, years)
    """

    # --- Color-blind-friendly palette (CUD palette) for cohesion ---
    cbf_colors = [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#D55E00',  # Vermillion
        '#CC79A7',  # Reddish Purple (extra)
        '#0072B2'   # Blue (extra)
        ]


    # --- Critical slopes (deg) per cohesion (edit as needed) ---
    crit_slopes = {
        760: 28.0,
        1920: 29.0,
        # add more if needed, e.g.:
        # 6400: 27.0,
    }

    # Figure setup
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor("#f0f0f0")
    plt.grid(True, alpha=0.4)

    legend_entries = []

    # Unique cohesions and saturations
    unique_coh = sorted(combined_df["Cohesion"].unique(), reverse=True)
    unique_m = sorted(combined_df["m"].unique())

    # Map cohesion → color
    color_map = {coh: cbf_colors[i % len(cbf_colors)] for i, coh in enumerate(unique_coh)}

    # Map saturation m → linestyle & marker
    # (Edit these if you add more m values)
    linestyle_map = {
        0.85: '--',
        1.00: '-',
    }
    marker_map = {
        0.85: 'o',
        1.00: 's',
    }

    # Group by (Cohesion, m) so we fit & plot each combo separately
    grouped = combined_df.groupby(["Cohesion", "m"])

    for (cohesion, m_val), group in sorted(grouped, key=lambda x: (x[0][0], x[0][1]), reverse=True):
        x = group["Avg_Slope_deg"].values
        y = group["Year"].values

        color = color_map[cohesion]
        linestyle = linestyle_map.get(float(m_val), '-.')
        marker = marker_map.get(float(m_val), 'o')

        # --- Scatter points ---
        plt.scatter(
            x, y,
            alpha=0.8,
            color=color,
            s=60,
            marker=marker,
        )

        # Decide which inverse model to use
        theta_crit = crit_slopes.get(cohesion, None)

        if theta_crit is not None:
            # ---- BOUNDED CASE: RI = a / (θ - θ_crit) ----
            # Fit only on data above (theta_crit + small buffer)
            fit_mask = x > (theta_crit + 0.1)
            xdata = x[fit_mask]
            ydata = y[fit_mask]

            if len(xdata) > 2:
                def model_for_fit(x_local, a):
                    return inverse_model_fixed(x_local, a, theta_crit)

                popt, _ = curve_fit(model_for_fit, xdata, ydata, p0=[1000])
                a_fit = popt[0]

                # R² for this (Cohesion, m) combo
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
                print(
                    f"⚠️ Not enough points above θ_crit={theta_crit:.1f}° "
                    f"to fit C={cohesion} Pa, m={m_val:.2f}"

                )

        else:
            # ---- UNBOUNDED CASE: RI = a / (θ + b) ----
            if len(x) > 2:
                popt, _ = curve_fit(inverse_model, x, y, p0=[1000, -25])
                a_fit, b_fit = popt

                y_pred = inverse_model(x, *popt)
                residuals = y - y_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot)

                x_fit = np.linspace(min(x) - 0.5, max(x) + 1, 300)
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
                print(
                    f"⚠️ Not enough points to fit C={cohesion} Pa, m={m_val:.2f} "
                    "with unbounded model"
                    )

    # ---- Axis & legend formatting ----
    plt.yscale("log")
    plt.xlabel("Hollow Slope, $\\theta_H$ (°)", fontsize=14, fontweight='bold')
    plt.ylabel("Recurrence Interval, RI (years)", fontsize=14, fontweight='bold')

    if legend_entries:
        handles, labels = zip(*legend_entries)
        legend = plt.legend(
            handles, labels,
            title="Cohesion & m Fit Equation",
            fontsize=11,
            title_fontsize=12,
            loc='upper right',
            bbox_to_anchor=(0.98, 1.01),
            frameon=True
        )
        legend.get_frame().set_linewidth(3)

    tick_range = np.arange(
        combined_df["Avg_Slope_deg"].min() // 3 * 3,
        combined_df["Avg_Slope_deg"].max() + 1,
        3
    )
    plt.xticks(tick_range)

    plt.tight_layout()
    plt.show()


# -------- Example usage --------
# If you've read your Excel into combined_df like:
# combined_df = pd.read_excel("combined_df.xlsx")
# and it already has Cohesion, m, Avg_Slope_deg, Year:

# plot_ri_vs_slope_by_cohesion_and_saturation(combined_df)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# 1. LOAD YOUR EXCEL FILE
# ============================================================

combined_df = pd.read_excel(r"C:\Users\sdavilao\Documents\combined_df.xlsx")

print("Loaded rows:", len(combined_df))
print("Columns:", combined_df.columns.tolist())


# ============================================================
# 2. INVERSE MODELS
# ============================================================

def inverse_model(x, a, b):
    """
    Unbounded inverse model: RI = a / (x + b)
    """
    return a / (x + b)

def inverse_model_fixed(x, a, theta_crit):
    """
    Bounded inverse model: RI = a / (x - theta_crit)
    with vertical asymptote at theta_crit (deg).
    """
    return a / (x - theta_crit)


# ============================================================
# 3. PLOT FUNCTION: BY COHESION (COLOR) AND SATURATION (STYLE)
# ============================================================

def plot_ri_vs_slope_by_cohesion_and_saturation(df):
    """
    df must contain at least:
      - 'Cohesion'
      - 'Saturation'  (this is m)
      - 'Avg_Slope_deg'
      - 'Year'
    """

    # Drop any rows with missing key fields
    df = df.dropna(subset=["Avg_Slope_deg", "Year", "Cohesion", "Saturation"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#f0f0f0")
    ax.grid(True)

    # Unique cohesion and saturation values
    unique_coh = sorted(df["Cohesion"].unique())
    unique_sat = sorted(df["Saturation"].unique())

    # Color-blind-friendly palette (CUD)
    cbf_colors = [
        '#D55E00',  # Vermillion
        '#009E73',  # Bluish Green
        '#56B4E9',  # Sky Blue
        '#E69F00',  # Orange
        '#CC79A7',  # Reddish Purple
        '#0072B2'   # Blue
    ]

    # Map cohesion -> color using the CUD palette
    color_map = {coh: cbf_colors[i % len(cbf_colors)] for i, coh in enumerate(unique_coh)}

    # Line style / marker per saturation
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    ls_map = {}
    mk_map = {}
    for i, sat in enumerate(unique_sat):
        ls_map[sat] = linestyles[i % len(linestyles)]
        mk_map[sat] = markers[i % len(markers)]

    # Critical slopes (deg) ONLY for m = 0.85
    crit_slopes = {
        760: 28.0,
        1920: 29.0,
    }

    legend_handles = []
    legend_labels = []

    # Group by (Cohesion, Saturation)
    grouped = df.groupby(["Cohesion", "Saturation"])

    for (coh, sat), g in grouped:
        x = g["Avg_Slope_deg"].values
        y = g["Year"].values

        color = color_map[coh]
        ls = ls_map[sat]
        mk = mk_map[sat]

        # Scatter points
        ax.scatter(x, y, color=color, marker=mk, s=50, alpha=0.8)

        # Decide if this combo should be bounded
        use_bounded = (abs(float(sat) - 0.85) < 1e-6) and (coh in crit_slopes)
        theta_crit = crit_slopes[coh] if use_bounded else None

        # Only fit if we have enough points
        if len(x) > 2:
            try:
                if use_bounded:
                    # -------- BOUNDED MODEL: RI = a / (theta - theta_crit) --------
                    # Fit only for x > theta_crit + small buffer
                    fit_mask = x > (theta_crit + 0.1)
                    xdata = x[fit_mask]
                    ydata = y[fit_mask]

                    if len(xdata) < 3:
                        print(f"Warning: not enough points above {theta_crit}° for C={coh}, m={sat}")
                        continue

                    def model_for_fit(x_local, a):
                        return inverse_model_fixed(x_local, a, theta_crit)

                    popt, _ = curve_fit(model_for_fit, xdata, ydata, p0=[1000])
                    a_fit = popt[0]

                    # Smooth curve
                    x_fit = np.linspace(theta_crit + 0.1, max(x), 200)
                    y_fit = model_for_fit(x_fit, a_fit)

                    line, = ax.plot(x_fit, y_fit, color=color, linestyle=ls, lw=2)

                    label = (
                        f"C = {int(coh)} Pa, m = {sat:.2f}: "
                        f"RI = {a_fit:.1f} / (theta_H - {theta_crit:.1f})"
                    )
                    legend_handles.append(line)
                    legend_labels.append(label)

                else:
                    # -------- UNBOUNDED MODEL: RI = a / (theta + b) --------
                    popt, _ = curve_fit(inverse_model, x, y, p0=[1000, -25])
                    a_fit, b_fit = popt

                    x_fit = np.linspace(min(x), max(x), 200)
                    y_fit = inverse_model(x_fit, a_fit, b_fit)

                    line, = ax.plot(x_fit, y_fit, color=color, linestyle=ls, lw=2)

                    label = (
                        f"C = {int(coh)} Pa, m = {sat:.2f}: "
                        f"RI = {a_fit:.1f} / (theta_H + {b_fit:.2f})"
                    )
                    legend_handles.append(line)
                    legend_labels.append(label)

            except Exception as e:
                print("Fit failed for cohesion", coh, "saturation", sat, ":", e)

    # Axes
    ax.set_yscale("log")
    # ax.set_xlabel("Hollow slope, $\\theta_H$ (deg)")
    # ax.set_ylabel("Recurrence interval, RI (years)")

    # Legend
    # if legend_handles:
    #     ax.legend(legend_handles, legend_labels, fontsize=8, loc="upper right")

    # Nice x ticks
    xmin = df["Avg_Slope_deg"].min()
    xmax = df["Avg_Slope_deg"].max()
    tick_range = np.arange(27, df["Avg_Slope_deg"].max() + 1, 3)
    ax.set_xticks(tick_range)

    fig.tight_layout()
    plt.savefig("C:/Users/sdavilao/Documents/RI_coh_sat.png", dpi=450, bbox_inches='tight')
    plt.show()


# ============================================================
# 4. RUN THE PLOT
# ============================================================

plot_ri_vs_slope_by_cohesion_and_saturation(combined_df)

