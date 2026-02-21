import os
import glob
import re

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from rasterstats import zonal_stats

# =============================================================================
# USER INPUTS: SET EXTENT & POINT ID HERE
# =============================================================================
extent = "ext4"          # e.g., "ext7", "ext10", "ext12", ...
target_point_id = 1      # the Point_ID you want to analyze

print(f"Running analysis for extent = {extent}, Point_ID = {target_point_id}")

# =============================================================================
# PATHS
# =============================================================================
base_dir = r"C:\Users\sdavilao\Documents\newcodesoil"

shapefile_path   = os.path.join(base_dir, "reproj_shp", f"{extent}_32610.shp")
dem_path         = os.path.join(base_dir, "dem_smooth_m_warp.tif")
slope_path       = os.path.join(base_dir, "slope_smooth_m_warp.tif")
downslope_lines  = os.path.join(base_dir, "polylines", f"{extent}_lines.shp")

soil_depth_dir   = os.path.join(base_dir, "simulation_results", "new", "GeoTIFFs", "reproj_tif")
soil_depth_pattern = os.path.join(soil_depth_dir, f"{extent}_total_soil_depth_*yrs_32610.tif")

results_dir      = os.path.join(base_dir, "results", "new")
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# LOAD SHAPEFILES & FILTER TO TARGET POINT
# =============================================================================
gdf = gpd.read_file(shapefile_path)
line_gdf = gpd.read_file(downslope_lines)

# Normalize id column name
if "id" in gdf.columns:
    gdf = gdf.rename(columns={"id": "Point_ID"})
if "id" in line_gdf.columns:
    line_gdf = line_gdf.rename(columns={"id": "Point_ID"})

# Filter to the chosen Point_ID
gdf = gdf[gdf["Point_ID"] == target_point_id].copy()
line_gdf = line_gdf[line_gdf["Point_ID"] == target_point_id].copy()

if gdf.empty:
    raise ValueError(f"No point with Point_ID = {target_point_id} in {shapefile_path}")
if line_gdf.empty:
    print(f"⚠️ No downslope line for Point_ID = {target_point_id} in {downslope_lines}, slope will be NaN.")

# Match CRS with DEM
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)
    if line_gdf.crs != dem_crs:
        line_gdf = line_gdf.to_crs(dem_crs)

# =============================================================================
# SOIL DEPTH FILES
# =============================================================================
soil_depth_files = sorted(glob.glob(soil_depth_pattern))
if not soil_depth_files:
    raise FileNotFoundError(f"No soil depth rasters matching pattern:\n{soil_depth_pattern}")

print(f"Found {len(soil_depth_files)} soil depth rasters.")

# Buffers to test (m)
buffer_sizes = [3, 4, 5, 6, 7, 8, 9]

# =============================================================================
# OPTIONAL: QUICK MAP OF BUFFERS
# =============================================================================
soil_depth_map = [f for f in soil_depth_files if "100yrs" in f or "200yrs" in f]
if soil_depth_map:
    soil_depth_path = soil_depth_map[0]

    buffer_geoms = []
    for _, row in gdf.iterrows():
        point_id = row["Point_ID"]
        point_geom = row.geometry
        for buffer_distance in buffer_sizes:
            buffer = point_geom.buffer(buffer_distance)
            buffer_geoms.append({
                "geometry": buffer,
                "Point_ID": point_id,
                "Buffer_Size": buffer_distance
            })

    buffer_gdf = gpd.GeoDataFrame(buffer_geoms, crs=gdf.crs)

    with rasterio.open(soil_depth_path) as src:
        fig, ax = plt.subplots(figsize=(8, 8))
        show(src, ax=ax, cmap="terrain")

        buffer_gdf.plot(ax=ax, column="Buffer_Size", cmap="viridis",
                        alpha=0.3, legend=True)
        gdf.plot(ax=ax, color="black", markersize=10)

        # Label point
        for x, y, pid in zip(gdf.geometry.x, gdf.geometry.y, gdf["Point_ID"]):
            ax.text(x, y, str(pid), fontsize=8, ha="center", va="center")

        ax.set_title(f"{extent}: Buffer radii for Point_ID = {target_point_id}")
        plt.tight_layout()
        plt.show()
else:
    print("ℹ️ No 100yr/200yr soil depth raster available for buffer map.")

# =============================================================================
# SLOPE ALONG DOWNSLOPE LINE FOR THIS POINT
# =============================================================================
point_slope_dict = {}
point_slope_count = {}

for _, line_row in line_gdf.iterrows():
    pid = line_row["Point_ID"]
    try:
        zs = zonal_stats([line_row["geometry"]], slope_path,
                         stats=["count", "mean"], nodata=-9999)
        avg = zs[0]["mean"]
        count = zs[0]["count"]
        point_slope_dict[pid] = avg
        point_slope_count[pid] = count
        print(f"[OK] Slope for Point {pid}: {avg:.4f} from {count} pixels")
    except Exception as e:
        point_slope_dict[pid] = np.nan
        point_slope_count[pid] = 0
        print(f"[FAIL LINE SLOPE] Point {pid}: {e}")

# =============================================================================
# LOOP BUFFERS & YEARS, EXTRACT SOIL DEPTH
# =============================================================================
results = []

for _, row in gdf.iterrows():
    point_geom = row.geometry
    point_id = row["Point_ID"]

    for buffer_distance in buffer_sizes:
        buffer_geom = point_geom.buffer(buffer_distance)
        buffer_json = [buffer_geom.__geo_interface__]
        print(f"\n[CHECK] {extent}, Point {point_id}, Buffer {buffer_distance} m")

        avg_slope = point_slope_dict.get(point_id, np.nan)
        slope_count = point_slope_count.get(point_id, 0)
        print(f"    → Constant slope used: {avg_slope}, pixel count: {slope_count}")

        for soil_depth_tif in soil_depth_files:
            m = re.search(r"_(\d+)yrs.*\.tif$", soil_depth_tif)
            if m:
                year = int(m.group(1))
            else:
                continue

            try:
                print(f"  🔎 Opening soil raster: {os.path.basename(soil_depth_tif)}")
                with rasterio.open(soil_depth_tif) as soil_depth_raster:
                    soil_depth_image, _ = mask(soil_depth_raster, buffer_json, crop=True)
                    soil_depth_image = soil_depth_image[0]
                    nodata_val = soil_depth_raster.nodata

                    valid = soil_depth_image[
                        (soil_depth_image != nodata_val) &
                        (soil_depth_image > 0)
                    ]

                    if valid.size == 0:
                        print(f"  ⚠️ No valid soil depth at Year {year}, Buffer {buffer_distance}")
                        avg_soil_depth = np.nan
                    else:
                        avg_soil_depth = float(np.mean(valid))
                        print(f"  → Avg soil depth: {avg_soil_depth:.4f}")

            except Exception as e:
                print(f"[FAIL SOIL] Year {year}, Buffer {buffer_distance}: {e}")
                avg_soil_depth = np.nan
                continue

            results.append({
                "Extent": extent,
                "Point_ID": point_id,
                "Year": year,
                "Buffer_Size": buffer_distance,
                "Avg_Slope": avg_slope,
                "Avg_Soil_Depth": avg_soil_depth
            })
            print(f"✅ Stored: Year {year}, Buffer {buffer_distance}")

print(f"\n🔍 Total results collected: {len(results)}")
df = pd.DataFrame(results)
if df.empty:
    raise ValueError("No results collected; check rasters/buffers/overlap.")

df = df.sort_values(by=["Point_ID", "Year", "Buffer_Size"]).reset_index(drop=True)

# =============================================================================
# FACTOR OF SAFETY
# =============================================================================
Sc = 1.25
pw = 1000
ps = 1600
g = 9.81
yw = g * pw
ys = g * ps
phi = np.deg2rad(41)
m = 1
l = 10
w = 6.7
C0 = 3600
j = 0.8

def calculate_fs(row):
    hollow_rad = np.radians(row["Avg_Slope"])
    z = row["Avg_Soil_Depth"]

    if np.isnan(z) or np.isnan(hollow_rad) or z <= 0:
        return np.nan

    Crb = C0 * np.exp(-z * j)
    Crl = (C0 / (j * z)) * (1 - np.exp(-z * j))

    K0 = 1 - np.sin(hollow_rad)

    Kp = np.tan((np.deg2rad(45))+(phi/2))**2
    Ka = np.tan((np.deg2rad(45))-(phi/2))**2

    Frb = (Crb + (np.cos(hollow_rad)**2) * z * (ys - yw * m) * np.tan(phi)) * l * w
    Frc = (Crl + (K0 * 0.5 * z * (ys - yw * m**2) * np.tan(phi))) * \
          (np.cos(hollow_rad) * z * l * 2)
    Frddu = (Kp - Ka) * 0.5 * (z**2) * (ys - yw * (m**2)) * w
    Fdc = (np.sin(hollow_rad)) * (np.cos(hollow_rad)) * z * ys * l * w

    return (Frb + Frc + Frddu) / Fdc if Fdc != 0 else np.nan

df["FS"] = df.apply(calculate_fs, axis=1)

# =============================================================================
# FIND OPTIMAL BUFFER (FS = 1 EARLIEST) FOR THIS POINT
# =============================================================================
from scipy.interpolate import interp1d
import numpy as np

# -----------------------------------------------------------------------------
# FIND OPTIMAL BUFFER (FS = 1 EARLIEST) FOR THIS POINT, IGNORING ALREADY-FAILED
# -----------------------------------------------------------------------------
point_data = df[df["Point_ID"] == target_point_id].copy()
buffer_first_crossings = {}

for buffer_size in sorted(point_data["Buffer_Size"].unique()):
    buffer_data = point_data[point_data["Buffer_Size"] == buffer_size].sort_values("Year")
    years     = buffer_data["Year"].values
    fs_values = buffer_data["FS"].values

    # Need at least two FS values and some finite ones
    if len(years) < 2 or np.all(~np.isfinite(fs_values)):
        continue

    # 🔴 Skip buffers that are already failed at the first time step
    if fs_values[0] <= 1.0:
        print(f"[SKIP] Buffer {buffer_size}: FS(Year={years[0]})={fs_values[0]:.3f} ≤ 1 at start")
        continue

    # Look for sign changes of (FS - 1) between consecutive samples
    diff = fs_values - 1.0
    idx = np.where(
        np.isfinite(diff[:-1]) &
        np.isfinite(diff[1:]) &
        (diff[:-1] * diff[1:] <= 0)
    )[0]

    if idx.size == 0:
        # never crosses 1 (stays >1 or <1 but not from the start)
        continue

    # Use the first crossing segment
    i = idx[0]
    fs_i, fs_ip1 = fs_values[i],   fs_values[i+1]
    t_i,  t_ip1  = years[i],       years[i+1]

    if np.isclose(fs_ip1, fs_i):
        frac = 0.0
    else:
        frac = (1.0 - fs_i) / (fs_ip1 - fs_i)

    estimated_year = t_i + frac * (t_ip1 - t_i)

    if years.min() < estimated_year < years.max():
        buffer_first_crossings[buffer_size] = estimated_year
        print(f"[FS=1] Buffer {buffer_size} → Year ~ {estimated_year:.2f}")

# -----------------------------------------------------------------------------
# Choose optimal buffer and interpolate soil depth & slope
# -----------------------------------------------------------------------------
if not buffer_first_crossings:
    print(f"⚠️ No FS=1 crossings found for {extent}, Point_ID {target_point_id} "
          f"(ignoring already-failed buffers).")
    optimal_buffer_size = None
    optimal_year = None
else:
    optimal_buffer_size = min(buffer_first_crossings, key=buffer_first_crossings.get)
    optimal_year = buffer_first_crossings[optimal_buffer_size]
    print(f"\n⭐ Optimal buffer: {optimal_buffer_size} m (FS=1 at ~{optimal_year:.2f} yrs)")

    selected = point_data[point_data["Buffer_Size"] == optimal_buffer_size].sort_values("Year")

    soil_depth_interp = interp1d(
        selected["Year"].values,
        selected["Avg_Soil_Depth"].values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    slope_interp = interp1d(
        selected["Year"].values,
        selected["Avg_Slope"].values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    estimated_soil_depth = float(soil_depth_interp(optimal_year))
    estimated_slope      = float(slope_interp(optimal_year))

    df_optimal_interpolated = pd.DataFrame([{
        "Extent": extent,
        "Point_ID": target_point_id,
        "Optimal_Buffer_m": optimal_buffer_size,
        "Year": optimal_year,
        "FS": 1.0,
        "Avg_Soil_Depth_m": estimated_soil_depth,
        "Avg_Slope_deg": estimated_slope
    }])

    print("\nOptimal buffer & FS=1 interpolation:")
    print(df_optimal_interpolated)

    # out_csv = os.path.join(
    #     results_dir,
    #     f"optimal_buffer_results_interpolated_{extent}_760_PointID{target_point_id}.csv"
    # )
    # df_optimal_interpolated.to_csv(out_csv, index=False)
    # print(f"\nSaved optimal result to:\n{out_csv}")


# =============================================================================
#%% PLOT: DOUBLE Y-AXIS (FS & SOIL DEPTH VS TIME) FOR OPTIMAL BUFFER
# =============================================================================
if optimal_buffer_size is not None:
    plot_data = point_data[point_data["Buffer_Size"] == optimal_buffer_size].sort_values("Year")
    title_suffix = f"(Optimal buffer = {optimal_buffer_size} m)"
else:
    # fallback: smallest buffer if no FS=1 crossing
    b0 = sorted(point_data["Buffer_Size"].unique())[0]
    plot_data = point_data[point_data["Buffer_Size"] == b0].sort_values("Year")
    title_suffix = f"(Fallback buffer = {b0} m; no FS=1 crossing)"

years_plot = plot_data["Year"].values
fs_plot = plot_data["FS"].values
sd_plot = plot_data["Avg_Soil_Depth"].values

fig, ax1 = plt.subplots(figsize=(7, 5))
# --- Light gray background for the PLOT AREA only ---
ax1.set_facecolor("#f0f0f0")     # light gray

# --- Grid lines ---
#ax1.grid(True, which='both', color="#d0d0d0", linewidth=1, alpha=0.7)

# Left y: FS
ax1.plot(years_plot, fs_plot, marker="o", lw=1.2, color="C0", label="FS")
# ax1.axhline(1.0, ls="--", lw=1.2, color="red", alpha=0.9, label="FS = 1")
ax1.set_xlabel("Year")
ax1.set_ylabel("FS", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")

if np.isfinite(fs_plot).any():
    fs_min = np.nanmin(fs_plot)
    fs_max = np.nanmax(fs_plot)
    pad = 0.1 * max(1e-9, fs_max - fs_min)
    ax1.set_ylim(fs_min - pad, fs_max + pad)

# Right y: Soil depth
ax2 = ax1.twinx()
ax2.set_facecolor("#f0f0f0")   # match the background
ax2.plot(years_plot, sd_plot, marker="s", lw=1.2, color="C1", label="Avg Soil Depth")
ax2.set_ylabel("Average Soil Depth (m)", color="C1")
ax2.tick_params(axis="y", labelcolor="C1")
ax2.set_ylim(0, 0.75)

# Mark FS=1 crossing if we have it
if optimal_buffer_size is not None and optimal_year is not None:
    # ax1.axvline(optimal_year, ls=":", lw=2, color="red", alpha=0.9, label="FS = 1 crossing")
    
    # #point on the FS curve at FS = 1
    # ax1.plot(
    #     [optimal_year], [1.0],
    #     marker="x",   # hollow marker (remove this if you want it filled)
    #     markeredgecolor="red",
    #     markersize=15,
    #     zorder=6,
    #     label="FS = 1 point"
    # )
    # ax1.text(
    #     optimal_year,
    #     ax1.get_ylim()[1],
    #     f"Year = {optimal_year:.1f}",
    #     ha='center', va='bottom',
    #     fontsize=9, color='k', fontweight='bold'
    # )
    # ax2.plot([optimal_year], [estimated_soil_depth], "x", markersize=15, color="red", zorder=5)
    # ax2.annotate(
    #     f"Depth @ FS=1 = {estimated_soil_depth:.2f} m\nYear ≈ {optimal_year:.1f}",
    #     xy=(optimal_year, estimated_soil_depth),
    #     xytext=(8, 10),
    #     textcoords="offset points",
    #     bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.5")
    # )
    #zoom to just after failure
    zoom_factor = 1.2
    t_max = min(optimal_year * zoom_factor, years_plot.max())
    ax1.set_xlim(0, t_max)
else:
    ax1.set_xlim(0, max(1000, years_plot.max() * 1.05))

plt.title(f"{extent} – Point_ID {target_point_id}\nFS and Avg Soil Depth vs Time {title_suffix}")

# Combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
#fig.legend(h1 + h2, l1 + l2, loc="upper right")

plt.tight_layout()
plt.savefig("C:/Users/sdavilao/Documents/FSSD_T.png", dpi=450, bbox_inches='tight')
plt.show()


#%% PLOT: FS vs Time, colored by Buffer Size (Matplotlib)
import matplotlib.pyplot as plt
import numpy as np

plot_df = point_data.sort_values(["Buffer_Size", "Year"]).copy()

fig, ax = plt.subplots(figsize=(7, 5))
# --- Light gray background for the PLOT AREA only ---
ax.set_facecolor("#f0f0f0")     # light gray

buffer_sizes = sorted(plot_df["Buffer_Size"].unique())
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(buffer_sizes)))

for color, b in zip(colors, buffer_sizes):
    sub = plot_df[plot_df["Buffer_Size"] == b]
    ax.plot(
        sub["Year"], sub["FS"],
        marker="o",
        markersize=6,
        lw=1.7,
        color=color,
        label=f"Buffer {b} m"
    )

# FS = 1 line
# ax.axhline(1.0, ls="--", lw=1.5, color="red", alpha=0.8, label="FS = 1")

# Labels
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Factor of Safety (FS)", fontsize=12)
ax.set_title(f"{extent} – Point_ID {target_point_id}\nFS vs Time by Buffer Size", fontsize=13)

# Legend outside or inside?
#ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

# Optional zoom to region around failure
if optimal_year is not None:
    ax.set_xlim(0, optimal_year * 1.2)


# ax.set_xlim(1110, 1130)
# ax.set_ylim(0.9975, 1.01)

# Light grid
ax.grid(True, which="both", color="#d0d0d0", alpha=0.7)

plt.tight_layout()
#plt.savefig("C:/Users/sdavilao/Documents/FS_T_Buff_in.png", dpi=450, bbox_inches='tight')
plt.show()

#%% PLOT: FS vs Buffer Size at Failure Year of Optimal Buffer
import numpy as np
import matplotlib.pyplot as plt

# Safety check
if optimal_year is None:
    raise ValueError("optimal_year is None – run the FS=1 / optimal buffer code first.")

# Get buffers and sort them
buffers = sorted(point_data["Buffer_Size"].unique())

fs_at_failure = []
for b in buffers:
    sub = point_data[point_data["Buffer_Size"] == b].sort_values("Year")

    # Interpolate FS for this buffer at the optimal failure year
    years_b = sub["Year"].values
    fs_b    = sub["FS"].values

    # np.interp assumes increasing x; we sorted above
    fs_interp_b = np.interp(optimal_year, years_b, fs_b)
    fs_at_failure.append(fs_interp_b)

fs_at_failure = np.array(fs_at_failure)

fig, ax = plt.subplots(figsize=(7, 5))

# Light gray background, no grid
ax.set_facecolor("#f0f0f0")
ax.grid(False)

# Base line: FS vs buffer size at failure year
ax.plot(
    buffers, fs_at_failure,
    marker="o",
    markersize=8,
    lw=2,
    color="C0",
    label=f"FS at Year ≈ {optimal_year:.1f}"
)

# Horizontal FS=1 line
# ax.axhline(1.0, ls="--", lw=1.2, color="red", alpha=0.8, label="FS = 1")

# Highlight the optimal buffer explicitly
if optimal_buffer_size is not None:
    # Find its FS at failure year
    fs_opt = fs_at_failure[buffers.index(optimal_buffer_size)]
    ax.plot(
        [optimal_buffer_size], [fs_opt],
        marker="o",
        markersize=10,
        markerfacecolor="red",
        markeredgecolor="k",
        zorder=5,
        label=f"Optimal buffer = {optimal_buffer_size} m"
    )

ax.set_xlabel("Buffer Size (m)", fontsize=12)
ax.set_ylabel(f"FS at Year ≈ {optimal_year:.1f}", fontsize=12)
# ax.set_title(f"{extent} – Point_ID {target_point_id}\nFS vs Buffer Size at Failure Year of Optimal Buffer",
#              fontsize=13)

#ax.legend(loc="best")
plt.tight_layout()
plt.savefig("C:/Users/sdavilao/Documents/FS_Buff.png", dpi=450, bbox_inches='tight')
plt.show()

#%% PLOT: FS & Soil Depth vs Buffer Size at Failure Year of Optimal Buffer
import numpy as np
import matplotlib.pyplot as plt

# Safety check
if optimal_year is None:
    raise ValueError("optimal_year is None – run the FS=1 / optimal buffer code first.")

# Get buffers and sort them
buffers = sorted(point_data["Buffer_Size"].unique())

fs_at_failure = []
sd_at_failure = []

for b in buffers:
    sub = point_data[point_data["Buffer_Size"] == b].sort_values("Year")

    years_b = sub["Year"].values
    fs_b    = sub["FS"].values
    sd_b    = sub["Avg_Soil_Depth"].values

    # Interpolate FS and soil depth at the optimal failure year
    fs_interp_b = np.interp(optimal_year, years_b, fs_b)
    sd_interp_b = np.interp(optimal_year, years_b, sd_b)

    fs_at_failure.append(fs_interp_b)
    sd_at_failure.append(sd_interp_b)

fs_at_failure = np.array(fs_at_failure)
sd_at_failure = np.array(sd_at_failure)

fig, ax1 = plt.subplots(figsize=(7, 5))

# Light gray background, no grid
ax1.set_facecolor("#f0f0f0")
ax1.grid(False)

# ---------------------------
# Left axis: FS vs buffer size
# ---------------------------
ax1.plot(
    buffers, fs_at_failure,
    marker="o",
    markersize=7,
    lw=2,
    color="C0",
    label=f"FS at Year ≈ {optimal_year:.1f}"
)

#ax1.set_xlabel("Buffer Size (m)", fontsize=12)
#ax1.set_ylabel(f"FS at Year ≈ {optimal_year:.1f}", fontsize=12, color="C0")
ax1.tick_params(axis="y", labelcolor="C0")

# Optional FS=1 line
# ax1.axhline(1.0, ls="--", lw=1.2, color="red", alpha=0.8, label="FS = 1")

# Highlight the optimal buffer on FS curve
if optimal_buffer_size is not None:
    fs_opt = fs_at_failure[buffers.index(optimal_buffer_size)]
    ax1.plot(
        [optimal_buffer_size], [fs_opt],
        marker="o",
        markersize=10,
        markerfacecolor="red",
        markeredgecolor="k",
        zorder=5,
        label=f"Optimal buffer = {optimal_buffer_size} m"
    )

# -------------------------------
# Right axis: Soil depth vs buffer
# -------------------------------
ax2 = ax1.twinx()
ax2.set_facecolor("#f0f0f0")  # match

ax2.plot(
    buffers, sd_at_failure,
    marker="s",
    markersize=7,
    lw=2,
    linestyle="--",
    color="C1",
    label=f"Soil depth at Year ≈ {optimal_year:.1f}"
)

#ax2.set_ylabel("Average Soil Depth (m)", fontsize=12, color="C1")
ax2.tick_params(axis="y", labelcolor="C1")

# If you want to cap soil depth, uncomment:
# ax2.set_ylim(0, 0.8)

# Highlight optimal buffer on soil-depth curve too
if optimal_buffer_size is not None:
    sd_opt = sd_at_failure[buffers.index(optimal_buffer_size)]
    ax2.plot(
        [optimal_buffer_size], [sd_opt],
        marker="s",
        markersize=10,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
        zorder=6,
        label=f"Depth @ optimal buffer ({sd_opt:.2f} m)"
    )

# ---------------------------
# Combined legend
# ---------------------------
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

fig.legend(
    h1 + h2, l1 + l2,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=8
)

plt.tight_layout()
plt.savefig("C:/Users/sdavilao/Documents/FSSD_OptBuff_withDepth.png", dpi=450, bbox_inches='tight')
plt.show()

#%% PLOT: FS & Soil Depth vs Buffer Size at Failure Year, colored by buffer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D   # <-- For custom legend handles

# Safety check
if optimal_year is None:
    raise ValueError("optimal_year is None – run the FS=1 / optimal buffer code first.")

buffers = sorted(point_data["Buffer_Size"].unique())

fs_at_failure = []
sd_at_failure = []

for b in buffers:
    sub = point_data[point_data["Buffer_Size"] == b].sort_values("Year")

    years_b = sub["Year"].values
    fs_b    = sub["FS"].values
    sd_b    = sub["Avg_Soil_Depth"].values

    fs_interp_b = np.interp(optimal_year, years_b, fs_b)
    sd_interp_b = np.interp(optimal_year, years_b, sd_b)

    fs_at_failure.append(fs_interp_b)
    sd_at_failure.append(sd_interp_b)

fs_at_failure = np.array(fs_at_failure)
sd_at_failure = np.array(sd_at_failure)


# ---- SETUP VIRIDIS COLOR RANGE ----
vmin = min(buffers)
vmax = max(buffers)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.get_cmap("viridis")


fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.set_facecolor("#f0f0f0")
ax1.grid(False)

# ---------------------------
# Left axis: FS curve, colored per buffer
# ---------------------------
for b, fs_val in zip(buffers, fs_at_failure):
    c = cmap(norm(b))
    ax1.plot(b, fs_val, marker="o", markersize=10, color=c)

# Faint connecting line (solid)
ax1.plot(buffers, fs_at_failure, lw=1.8, color="black", alpha=0.35)

#ax1.set_xlabel("Buffer Size (m)", fontsize=12)
#ax1.set_ylabel(f"FS at Year ≈ {optimal_year:.1f}", fontsize=12, color="black")
ax1.tick_params(axis="y", labelcolor="black")


# -------------------------------
# Right axis: Soil depth curve (same colors)
# -------------------------------
ax2 = ax1.twinx()
ax2.set_facecolor("#f0f0f0")

for b, sd_val in zip(buffers, sd_at_failure):
    c = cmap(norm(b))
    ax2.plot(b, sd_val, marker="s", markersize=10, color=c)

# Faint connecting line (dashed)
ax2.plot(buffers, sd_at_failure, lw=1.8, color="black", alpha=0.35, linestyle="--")

#ax2.set_ylabel("Average Soil Depth (m)", fontsize=12, color="#35B779")
ax2.tick_params(axis="y", labelcolor="#35B779")


# -------------------------------
# Highlight optimal buffer
# -------------------------------
if optimal_buffer_size is not None:
    i_opt = buffers.index(optimal_buffer_size)
    c_opt = cmap(norm(optimal_buffer_size))

    ax1.plot(
        buffers[i_opt], fs_at_failure[i_opt],
        marker="o", markersize=13,
        markeredgecolor="black",
        markerfacecolor=c_opt,
        zorder=5
    )

    ax2.plot(
        buffers[i_opt], sd_at_failure[i_opt],
        marker="s", markersize=13,
        markeredgecolor="black",
        markerfacecolor=c_opt,
        zorder=6
    )


# -------------------------------
# Colorbar (for buffer sizes)
# -------------------------------
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax1, pad=0.1)
# cbar.set_label("Buffer Size (m)")


# -------------------------------
# NEW: Legend for line styles
# -------------------------------
# legend_elements = [
#     Line2D([0], [0], color="black", lw=2, alpha=0.35, linestyle="-",
#            label="FS trend line"),
#     Line2D([0], [0], color="black", lw=2, alpha=0.35, linestyle="--",
#            label="Soil depth trend line"),
# ]

# fig.legend(
#     handles=legend_elements,
#     loc="lower right",
#     fontsize=9,
#     frameon=False
# )

plt.tight_layout()
plt.savefig("C:/Users/sdavilao/Documents/FSSD_Buff_T_1.png", dpi=450, bbox_inches='tight')
plt.show()

#%% PLOT: FS vs Buffer Size at 400, 500, and Failure Year (Viridis + Line Styles)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Safety check
if optimal_year is None:
    raise ValueError("optimal_year is None – run FS=1 / optimal buffer code first.")

# Times to sample
times_to_plot  = [1090, 1100, optimal_year]
time_labels    = ["400 yrs", "500 yrs", f"Failure year ≈ {optimal_year:.0f} yrs"]
line_styles    = ["--", ":", "-"]

# Buffers sorted
buffers = sorted(point_data["Buffer_Size"].unique())

# Viridis colormap for buffer markers
cmap = cm.viridis
buffer_colors = cmap(np.linspace(0, 1, len(buffers)))

# Compute FS for each time across all buffers
fs_by_time = {}

for t in times_to_plot:
    fs_at_t = []
    for b in buffers:
        sub = point_data[point_data["Buffer_Size"] == b].sort_values("Year")
        years_b = sub["Year"].values
        fs_b    = sub["FS"].values

        # Only interpolate where valid
        if len(years_b) < 2 or t < years_b.min() or t > years_b.max():
            fs_at_t.append(np.nan)
        else:
            fs_at_t.append(np.interp(t, years_b, fs_b))

    fs_by_time[t] = np.array(fs_at_t)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_facecolor("#f0f0f0")
ax.grid(
    True,
    which="both",
    color="#d0d0d0",
    alpha=0.7
)

# Plot each time curve as a full line so linestyle becomes visible
for (t, lab, ls) in zip(times_to_plot, time_labels, line_styles):
    y = fs_by_time[t]
    valid = np.isfinite(y)

    # Plot the line (line style now works!)
    ax.plot(
        np.array(buffers)[valid],
        y[valid],
        linestyle=ls,
        linewidth=2,
        color="black",  # keep the lines neutral so color = buffer
        label=f"FS at {lab}"
    )

    # Add markers colored by buffer
    for i, (b, fs_val) in enumerate(zip(buffers, y)):
        if np.isfinite(fs_val):
            ax.plot(
                b, fs_val,
                marker="o",
                markersize=8,
                color=buffer_colors[i],
            )

# --- Highlight optimal buffer at failure year ---
fail_t = times_to_plot[-1]
y_fail = fs_by_time[fail_t]

# if optimal_buffer_size in buffers:
#     i_opt = buffers.index(optimal_buffer_size)
#     fs_opt = y_fail[i_opt]
#     if np.isfinite(fs_opt):
#         ax.plot(
#             optimal_buffer_size, fs_opt,
#             marker="o",
#             markersize=12,
#             markerfacecolor="none",
#             markeredgecolor="red",
#             markeredgewidth=2.5,
#             zorder=5,
#             label=f"Optimal buffer = {optimal_buffer_size} m"
#         )

# Labels
# ax.set_xlabel("Buffer Size (m)", fontsize=12)
# ax.set_ylabel("Factor of Safety (FS)", fontsize=12)
# ax.set_title(
#     f"{extent} – Point_ID {target_point_id}\nFS vs Buffer Size (400, 500, Failure Year)",
#     fontsize=13
# )

# # Legend
# ax.legend(loc="best", fontsize=9)

plt.tight_layout()
plt.savefig("C:/Users/sdavilao/Documents/FS_Buff_T.png", dpi=450, bbox_inches='tight')
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Filter to this point's data
point_data = df[df["Point_ID"] == target_point_id].copy()

# Get the list of buffers available
buffers = sorted(point_data["Buffer_Size"].unique())

plt.figure(figsize=(8,6))
ax = plt.gca()

# Light gray background to match your other plots
ax.set_facecolor("#f0f0f0")

for b in buffers:
    buf_data = (
        point_data[point_data["Buffer_Size"] == b]
        .sort_values("Year")
    )

    if buf_data.empty:
        continue

    years = buf_data["Year"].values
    sd    = buf_data["Avg_Soil_Depth"].values

    ax.plot(
        years, sd,
        marker="o",
        lw=1.5,
        label=f"{b} m buffer"
    )

# Axis labels & style
ax.set_xlabel("Year")
ax.set_ylabel("Average Soil Depth (m)")
ax.set_title(f"{extent} – Point_ID {target_point_id}\nSoil Depth vs Time for All Buffers")

# Start soil depth at zero
ymin = 0.0
ymax = max(2, 1.05 * np.nanmax(point_data["Avg_Soil_Depth"]))
ax.set_ylim(ymin, ymax)

# Optional: limit x-axis to match soil evolution range
ax.set_xlim(0, point_data["Year"].max())

ax.legend(title="Buffer Size")
plt.tight_layout()
plt.show()

 #%% PLOT: FS & Soil Depth vs Time by Buffer, with Failure Markers
import matplotlib.pyplot as plt
import numpy as np

plot_df = point_data.sort_values(["Buffer_Size", "Year"]).copy()

fig, ax1 = plt.subplots(figsize=(7, 5))

# --- Light gray background for the PLOT AREA only ---
ax1.set_facecolor("#f0f0f0")     # light gray

buffer_sizes = sorted(plot_df["Buffer_Size"].unique())
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(buffer_sizes)))

# =======================
# Left axis: FS vs Time
# =======================
for color, b in zip(colors, buffer_sizes):
    sub = plot_df[plot_df["Buffer_Size"] == b].sort_values("Year")

    # Make the optimal buffer line a bit thicker
    lw = 2.0 if (optimal_buffer_size is not None and b == optimal_buffer_size) else 1.5

    ax1.plot(
        sub["Year"], sub["FS"],
        marker="o",
        markersize=7,
        lw=lw,
        color=color,
        label=f"Buffer {b} m – FS"
    )

# FS = 1 reference line
#ax1.axhline(1.0, ls="--", lw=1.3, color="red", alpha=0.7, label="FS = 1")

#ax1.set_xlabel("Year", fontsize=12)
#ax1.set_ylabel("Factor of Safety (FS)", fontsize=12, color="black")
ax1.tick_params(axis="y", labelcolor="black")
# ax1.set_title(
#     f"{extent} – Point_ID {target_point_id}\nFS & Soil Depth vs Time by Buffer Size",
#     fontsize=13
# )



# Optional zoom to region around failure
if "optimal_year" in locals() and optimal_year is not None:
    ax1.set_xlim(0, optimal_year * 1.2)
else:
    ax1.set_xlim(0, plot_df["Year"].max() * 1.05)

ax1.set_xlim(0, 1200)

# Light grid on FS axis
ax1.grid(True, which="both", color="#d0d0d0", alpha=0.7)

# ==========================
# Right axis: Soil Depth vs Time
# ==========================
ax2 = ax1.twinx()
ax2.set_facecolor("#f0f0f0")  # match background

for color, b in zip(colors, buffer_sizes):
    sub = plot_df[plot_df["Buffer_Size"] == b].sort_values("Year")

    lw = 2.0 if (optimal_buffer_size is not None and b == optimal_buffer_size) else 1.4

    ax2.plot(
        sub["Year"], sub["Avg_Soil_Depth"],
        marker="x",
        markersize=7,
        lw=lw,
        linestyle="--",
        color=color,
        label=f"Buffer {b} m – Depth"
    )

#ax2.set_ylabel("Average Soil Depth (m)", fontsize=12, color="black")
ax2.tick_params(axis="y", labelcolor="black")
ax2.set_ylim(0, max(2, 1.05 * np.nanmax(plot_df["Avg_Soil_Depth"])))

ax2.set_ylim(0, 0.8)
# ==========================
# Failure markers for optimal buffer
# ==========================
# if (optimal_buffer_size is not None and
#     "optimal_year" in locals() and optimal_year is not None and
#     "estimated_soil_depth" in locals() and estimated_soil_depth is not None):

#     # Vertical line at failure year
#     # ax1.axvline(optimal_year, ls=":", lw=2, color="red", alpha=0.9,
#     #             label="FS = 1 crossing")

#     # Marker on FS curve at FS = 1
#     # ax1.plot(
#     #     [optimal_year], [1.0],
#     #     marker="x",
#     #     markersize=12,
#     #     markeredgewidth=2,
#     #     markeredgecolor="red",
#     #     linestyle="None",
#     #     zorder=6,
#     #     label="FS = 1 point"
#     # )

#     # # Marker on soil-depth curve at depth @ failure
#     # ax2.plot(
#     #     [optimal_year], [estimated_soil_depth],
#     #     marker="x",
#     #     markersize=12,
#     #     markeredgewidth=2,
#     #     color="red",
#     #     linestyle="None",
#     #     zorder=6,
#     #     label=f"Depth @ FS=1 ({estimated_soil_depth:.2f} m)"
#     # )

# ==========================
# Combined legend
# ==========================
# h1, l1 = ax1.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()

# fig.legend(
#     h1 + h2, l1 + l2,
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5),
#     borderaxespad=0.0,
#     fontsize=8
# )

plt.tight_layout()
plt.savefig("C:/Users/sdavilao/Documents/FSSD_Buff_T.png", dpi=450, bbox_inches='tight')
plt.show()




