# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:28:14 2025

@author: sdavilao
"""

# -*- coding: utf-8 -*-
"""
Complete soil transport simulation with CRS-aware buffer masking and plotting.
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask
from osgeo import gdal
from landlab import imshowhs_grid
from landlab.components import TaylorNonLinearDiffuser
from landlab.io import read_esri_ascii, write_esri_ascii
import re

# Paths and constants
BASE_DIR = os.path.join(os.getcwd())
# Get all TIF files with the extX pattern
#tif_files = sorted(glob.glob(os.path.join(BASE_DIR, 'ext*.tif')))
INPUT_TIFF = 'ext26.tif'
POINTS_SHP = os.path.join(BASE_DIR, 'ext26.shp')  # shp file of center of hollow
BUFFER_SHP = os.path.join(BASE_DIR, 'ext26_buff.shp')  # shp file for buffer aroound hollow created by code
BUFFER_DISTANCE = 16 #in meters, and will give diameter of buffer
ft2mUS = 1200 / 3937
ft2mInt = 0.3048

OUT_DIR = os.path.join(BASE_DIR, 'simulation_results\\new\\new')
OUT_DIRpng = os.path.join(OUT_DIR, 'PNGs')
OUT_DIRtiff = os.path.join(OUT_DIR, 'GeoTIFFs')
OUT_DIRasc = os.path.join(OUT_DIR, 'ASCs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIRpng, exist_ok=True)
os.makedirs(OUT_DIRtiff, exist_ok=True)
os.makedirs(OUT_DIRasc, exist_ok=True)

def create_buffer_from_points(input_points_path, output_buffer_path, buffer_distance, target_crs=None):
    gdf = gpd.read_file(input_points_path)

    # Step 1: Ensure CRS is defined
    if gdf.crs is None:
        raise ValueError(f"❌ Shapefile {input_points_path} has no CRS. Please define it in QGIS or using gdf.set_crs().")

    # Step 2: Reproject to match DEM CRS
    if target_crs:
        if gdf.crs != target_crs:
            print(f"Reprojecting from {gdf.crs} to DEM CRS: {target_crs}")
            gdf = gdf.to_crs(target_crs)
    else:
        print("⚠️ No target CRS provided, using original shapefile CRS.")

    # Step 3: Buffer *after* reprojection
    buffer_geom = gdf.buffer(buffer_distance)
    buffered = gdf.copy()
    buffered["geometry"] = buffer_geom

    # Step 4: Save with updated CRS
    buffered.set_crs(gdf.crs, inplace=True)
    buffered.to_file(output_buffer_path)
    print(f"✅ Buffer created at {output_buffer_path}")
    print("Buffer bounds:", buffered.total_bounds)
    return output_buffer_path

def tiff_to_asc(in_path, out_path):
    with rasterio.open(in_path) as src:
        XYZunit = src.crs.linear_units if src.crs else "meters"
        mean_res = np.mean(src.res)
    gdal.Translate(out_path, in_path, format='AAIGrid', xRes=mean_res, yRes=mean_res)
    print(f"Converted GeoTIFF to ASCII with grid spacing {mean_res} ({XYZunit})")
    return mean_res, XYZunit

def asc_to_tiff(asc_path, tiff_path, meta):
    data = np.loadtxt(asc_path, skiprows=10)
    meta.update(dtype=rasterio.float32, count=1, compress='deflate')
    with rasterio.open(tiff_path, 'w', **meta) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    print(f"Saved GeoTIFF: {tiff_path}")

def apply_buffer_to_soil_depth(grid, shapefile, buffer_distance, dem_path):
    gdf = gpd.read_file(shapefile)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        transform = src.transform
        out_shape = src.read(1).shape

    if gdf.crs is None:
        print("⚠️  Shapefile has no CRS — assuming it aligns with DEM.")
    elif gdf.crs != dem_crs:
        print(f"Reprojecting shapefile from {gdf.crs} to match DEM CRS...")
        gdf = gdf.to_crs(dem_crs)

    buffered_geoms = gdf.buffer(buffer_distance)
    unified_geom = buffered_geoms.unary_union
    mask = geometry_mask([unified_geom], transform=transform, invert=True, out_shape=out_shape)

    if 'soil__depth' not in grid.at_node:
        soil_depth = np.full(grid.number_of_nodes, 0.5)
        grid.add_field('soil__depth', soil_depth, at='node')
    else:
        soil_depth = grid.at_node['soil__depth']

    soil_depth[np.flipud(mask).flatten()] = 0.0
    print(f"Applied 0m soil depth to buffer zones. Total zero-depth cells: {np.sum(soil_depth == 0.0)}")
    # Optional: show buffer mask and soil depth
    
    plt.figure(figsize=(6, 5))
    plt.imshow(mask, cmap='gray', origin='upper')
    plt.title("Buffer Mask (True = in buffer)")
    plt.colorbar(label="Mask Value")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(soil_depth.reshape(grid.shape), cmap='viridis', origin='upper')
    plt.title("Soil Depth After Masking")
    plt.colorbar(label="Soil Depth (m)")
    plt.tight_layout()
    plt.show()
    return grid

    # Optional: show mask and depth
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title("Buffer Mask (True = in buffer)")
    plt.show()

    plt.figure()
    plt.imshow(soil_depth.reshape(grid.shape), cmap='viridis')
    plt.title("Soil Depth After Masking")
    plt.colorbar(label="Soil Depth (m)")
    plt.show()

    return grid


def init_simulation(asc_file, K, Sc, XYZunit=None, shapefile=None, buffer_distance=BUFFER_DISTANCE, dem_path=None):
    grid, _ = read_esri_ascii(asc_file, name='topographic__elevation')
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    # Initialize with 0.5m soil depth
    soil_depth = np.full(grid.number_of_nodes, 0.5)
    grid.add_field('soil__depth', soil_depth, at='node')

    if shapefile and dem_path:
        grid = apply_buffer_to_soil_depth(grid, shapefile, buffer_distance, dem_path)

    if XYZunit is None or 'meter' in XYZunit.lower() or 'metre' in XYZunit.lower():
        Kc = K
    elif 'foot' in XYZunit.lower():
        Kc = K / (ft2mUS ** 2) if "US" in XYZunit else K / (ft2mInt ** 2)
    else:
        raise RuntimeError("Unsupported unit type for K conversion.")

    TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=Kc, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable="pass")
    return grid, TNLD

def plot_change(data, title, basefilename, time, K, grid_shape,vmin=None, vmax=None, cmap=None):
    if data.ndim == 1:
        data = data.reshape(grid_shape)
    plt.figure(figsize=(6, 5.25))
    plt.imshow(np.flipud(data), cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Value')
    plt.title(f"{title} at {time} yrs (K = {K})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    png_path = os.path.join(OUT_DIRpng, f"{basefilename}_{title.replace(' ', '_')}_{time}yrs_K{K}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

def save_as_tiff(data, filename, meta, grid_shape):
    if data.ndim == 1:
        data = data.reshape(grid_shape)

    meta = meta.copy()
    meta.update(
        dtype=rasterio.float32,
        count=1,
        compress='deflate'
    )

    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(np.flipud(data.astype(rasterio.float32)), 1)

    print(f"Saved TIFF: {filename}")


def plot_save(grid, z, basefilename, time, K, mean_res, XYZunit):
    plt.figure(figsize=(6, 5.25))
    imshowhs_grid(grid, z, plot_type="Hillshade")
    plt.title(f"{basefilename} Time {time} yrs (K = {K})")
    plt.tight_layout()
    png_path = os.path.join(OUT_DIRpng, f"{basefilename}_{time}yrs_K{K}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    asc_path = os.path.join(OUT_DIRasc, f"{basefilename}_{time}yrs_K{K}.asc")
    write_esri_ascii(asc_path, grid, names=['topographic__elevation'], clobber=True)
    return asc_path

def run_simulation(in_tiff, K, Sc, dt, target_time, point_shapefile):
    dem_path = os.path.join(BASE_DIR, in_tiff)

    # 🔁 Open the DEM and use its CRS when creating the buffer
    with rasterio.open(dem_path) as src:
        print(f"✅ DEM loaded from: {dem_path}")
        print("DEM bounds:", src.bounds)
        print("DEM CRS:", src.crs)
        print("DEM resolution:", src.res)
        buffer_shapefile = create_buffer_from_points(point_shapefile, BUFFER_SHP, BUFFER_DISTANCE, src.crs)

    basefilename = os.path.splitext(in_tiff)[0]
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(os.path.join(BASE_DIR, in_tiff), in_asc)

    grid, tnld = init_simulation(in_asc, K, Sc, XYZunit, shapefile=buffer_shapefile, dem_path=os.path.join(BASE_DIR, in_tiff))
    z_old = grid.at_node['topographic__elevation']
    soil_depth = grid.at_node['soil__depth']
    initial_soil_depth = soil_depth.copy()
    total_soil_depth = soil_depth.copy()

    with rasterio.open(os.path.join(BASE_DIR, in_tiff)) as src:
        meta = src.meta.copy()

    asc_path = plot_save(grid, z_old, basefilename, 0, K, mean_res, XYZunit)

    num_steps = int(target_time / dt)
    for i in range(num_steps):
        tnld.run_one_step(dt)
        time = (i + 1) * dt

        production_rate = (pr / ps) * (P0 * np.exp(-total_soil_depth / h0)) * dt
        z_new = grid.at_node['topographic__elevation']
        elevation_change = z_new - z_old
        if 'foot' in XYZunit.lower():
            elevation_change = elevation_change * ft2mUS  # or ft2mInt depending on your region

        nonzero_soil_mask = initial_soil_depth > 0.0
        erosion_exceeds = (np.abs(elevation_change) > initial_soil_depth) & nonzero_soil_mask
        if np.any(erosion_exceeds):
            z_new[erosion_exceeds] = z_old[erosion_exceeds] - total_soil_depth[erosion_exceeds]
            total_soil_depth[erosion_exceeds] = production_rate[erosion_exceeds]

        change_in_soil_depth = production_rate.copy()
        change_in_soil_depth[~erosion_exceeds] = elevation_change[~erosion_exceeds] + production_rate[~erosion_exceeds]
        total_soil_depth = np.where(erosion_exceeds, production_rate, total_soil_depth + change_in_soil_depth)

        grid.at_node['soil__depth'] = total_soil_depth
        z_old = z_new.copy()

        if time % 50 == 0:
            save_as_tiff(elevation_change, os.path.join(OUT_DIRtiff, f"{basefilename}_change_in_elevation_{time}yrs.tif"), meta, grid.shape)
            save_as_tiff(change_in_soil_depth, os.path.join(OUT_DIRtiff, f"{basefilename}_change_in_soil_depth_{time}yrs.tif"), meta, grid.shape)
            save_as_tiff(total_soil_depth, os.path.join(OUT_DIRtiff, f"{basefilename}_total_soil_depth_{time}yrs.tif"), meta, grid.shape)
            save_as_tiff(production_rate, os.path.join(OUT_DIRtiff, f"{basefilename}_production_rate_{time}yrs.tif"), meta, grid.shape)
            tiff_elevation_path = os.path.join(OUT_DIRtiff, f"{basefilename}_elevation_{time}yrs.tif")
            asc_to_tiff(asc_path, tiff_elevation_path, meta)

            grid_shape = grid.shape
            #plot_change(elevation_change, "Change in Elevation", basefilename, time, K, grid_shape, vmin=-1, vmax=1, cmap='coolwarm')
            #plot_change(change_in_soil_depth, "Change in Soil Depth", basefilename, time, K, grid_shape, vmin=-1, vmax=1, cmap='RdYlBu')
            plot_change(total_soil_depth, "Total Soil Depth", basefilename, time, K, grid_shape, vmin=0, vmax=5, cmap='viridis')
            #plot_change(production_rate, "Soil Produced", basefilename, time, K, grid_shape, vmin=0, vmax=0.01, cmap='plasma')


        if time % 50 == 0:
            asc_path = plot_save(grid, z_new, basefilename, time, K, mean_res, XYZunit)
            tiff_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs_K{K}.tif")
            asc_to_tiff(asc_path, tiff_path, meta)

    os.remove(in_asc)
    os.remove(in_asc.replace(".asc", ".prj"))
    print("Simulation complete.")

#%%
# Example run parameters
K = 0.0042  # Diffusion coefficient
Sc = 1.25  # Critical slope gradient
pr = 2000   # Ratio of production (example value)
ps = 1600   # Ratio of soil loss (example value)
P0 = 0.0003  # Initial soil production rate (example value, e.g., kg/m²/year)
h0 = 0.5   # Depth constant related to soil production (example value, e.g., meters)
dt = 50
target_time = 5000


run_simulation(INPUT_TIFF, K, Sc, dt, target_time,POINTS_SHP)

#%%
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import geometry_mask

# Reuse your globals:
# BASE_DIR, OUT_DIRtiff, OUT_DIRpng, INPUT_TIFF, BUFFER_SHP

TIFF_DIR = OUT_DIRtiff
base = os.path.splitext(INPUT_TIFF)[0]

# Files like: ext17_soil_produced_5000000yrs.tif
pattern_prod = re.compile(rf"{base}_production_rate_(\d+)yrs")


def build_buffer_mask(dem_path, buffer_shapefile):
    """
    Returns boolean mask (True = inside buffer) in the same orientation
    as arrays after np.flipud(read(1)).
    """
    with rasterio.open(dem_path) as src:
        transform = src.transform
        out_shape = src.read(1).shape
        dem_crs = src.crs

    gdf = gpd.read_file(buffer_shapefile)
    if gdf.crs is None:
        raise ValueError("Buffer shapefile has no CRS. Define it before using.")
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)

    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]

    mask_raw = geometry_mask(
        geoms,
        transform=transform,
        invert=True,   # True = inside shapes
        out_shape=out_shape
    )

    # You save TIFFs with np.flipud(data), so flip mask to match:
    mask = np.flipud(mask_raw)
    return mask


# Build buffer mask once
dem_path = os.path.join(BASE_DIR, INPUT_TIFF)
buffer_mask = build_buffer_mask(dem_path, BUFFER_SHP)


def load_soil_produced_buffered(path):
    """Load soil_produced TIFF, flip, apply buffer mask, handle nodata."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

    arr = np.flipud(arr)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    # Keep only inside buffer
    arr_masked = np.where(buffer_mask, arr, np.nan)
    return arr_masked


# --- Collect mean soil produced per timestep vs time ---
mean_prod = {}  # time (yrs) -> mean soil produced inside buffer

for fname in os.listdir(TIFF_DIR):
    m = pattern_prod.search(fname)
    if not m:
        continue

    t = int(m.group(1))  # time in yrs from filename
    fpath = os.path.join(TIFF_DIR, fname)

    arr = load_soil_produced_buffered(fpath)
    mean_prod[t] = np.nanmean(arr)

# --- Build time series ---
times = sorted(mean_prod.keys())
prod_ts = [mean_prod[t] for t in times]

# Cumulative soil produced (sum of per-step mean depth)
# Note: this is cumulative depth, not volume; for volume you’d multiply by cell area.
cum_prod_ts = np.cumsum(prod_ts)

# --- Plot ---
fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

# Per-step mean soil produced
ax[0].plot(times, prod_ts, marker='o', lw=1.8)
ax[0].set_ylabel("Mean soil produced\nper step (m)")
ax[0].set_title(f"Soil Produced Through Time\nBuffered Region – {base}")
ax[0].grid(True, linestyle='--', alpha=0.5)

# Cumulative soil produced
ax[1].plot(times, cum_prod_ts, marker='o', lw=1.8)
ax[1].set_ylabel("Cumulative mean soil\nproduced (m)")
ax[1].set_xlabel("Time (yrs)")
ax[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_ts = os.path.join(OUT_DIRpng, f"{base}_soil_produced_timeseries_BUFFERED.png")
plt.savefig(out_ts, dpi=150)
plt.close()

print("Saved soil-produced time series plot:", out_ts)

#%%
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import geometry_mask

# Reuse your existing globals:
# BASE_DIR, OUT_DIRtiff, OUT_DIRpng, INPUT_TIFF, BUFFER_SHP

TIFF_DIR = OUT_DIRtiff
base = os.path.splitext(INPUT_TIFF)[0]

# We expect filenames like:
#   ext17_change_in_elevation_5000000yrs.tif
pattern_elev = re.compile(rf"{base}_change_in_elevation_(\d+)yrs")


def build_buffer_mask(dem_path, buffer_shapefile):
    """
    Returns boolean mask (True = inside buffer) in the same orientation
    as arrays after np.flipud(read(1)).
    """
    with rasterio.open(dem_path) as src:
        transform = src.transform
        out_shape = src.read(1).shape
        dem_crs = src.crs

    gdf = gpd.read_file(buffer_shapefile)
    if gdf.crs is None:
        raise ValueError("Buffer shapefile has no CRS. Define it before using.")
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)

    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]

    mask_raw = geometry_mask(
        geoms,
        transform=transform,
        invert=True,    # True = inside shapes (buffer area)
        out_shape=out_shape
    )

    # Your save_as_tiff uses np.flipud(data), so flip mask to match TIFF orientation
    mask = np.flipud(mask_raw)
    return mask


# Build buffer mask once, based on the original DEM
dem_path = os.path.join(BASE_DIR, INPUT_TIFF)
buffer_mask = build_buffer_mask(dem_path, BUFFER_SHP)


def load_change_elev_buffered(path):
    """
    Load a change_in_elevation TIFF, flip vertically (to undo save_as_tiff flip),
    mask to buffer, convert nodata to NaN.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

    # Undo np.flipud used when writing
    arr = np.flipud(arr)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    # Only keep inside buffer
    arr_masked = np.where(buffer_mask, arr, np.nan)
    return arr_masked


# --- Collect mean net change in elevation vs time (inside buffer) ---
mean_dz = {}  # time (yrs) -> mean Δz (m), negative = net erosion

for fname in os.listdir(TIFF_DIR):
    m = pattern_elev.search(fname)
    if not m:
        continue

    t = int(m.group(1))   # time in yrs from filename
    fpath = os.path.join(TIFF_DIR, fname)

    dz = load_change_elev_buffered(fpath)
    mean_dz[t] = np.nanmean(dz)


# --- Build time series ---
times = sorted(mean_dz.keys())
dz_ts = [mean_dz[t] for t in times]

# Cumulative net change (depth-wise, mean over buffer)
cum_dz_ts = np.cumsum(dz_ts)

# ----------------------------
# Rates (m/yr) inside buffer
# ----------------------------
dt_model = dt  # reuse your simulation dt (50 yrs)

prod_rate_ts = np.array(prod_ts, dtype=float) / dt_model  # m/yr (since prod_ts is per-step increment)

dz_rate_ts = np.array(dz_ts, dtype=float) / dt_model      # signed net Δz rate (m/yr)

# "Erosion rate" as positive-down lowering only (ignore deposition / aggradation)
erosion_rate_ts = np.maximum(0.0, -dz_rate_ts)            # m/yr

# Optional: "deposition rate" (positive-up aggradation only)
deposition_rate_ts = np.maximum(0.0, dz_rate_ts)          # m/yr

plt.figure(figsize=(7, 4.5))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")
ax.set_xlim(0, 2500)

ax.plot(times, prod_rate_ts, marker='o', lw=1.8, label="Production rate (m/yr)")
ax.plot(times, erosion_rate_ts, marker='o', lw=1.8, label="Erosion rate (m/yr; lowering only)")

ax.set_xlabel("Time (yrs)")
ax.set_ylabel("Rate (m/yr)\n(mean inside buffer)")
ax.set_title(f"Production & Erosion Rates Through Time\nPoint buffered – {base}")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
out_png_rates = os.path.join(OUT_DIRpng, f"{base}_prod_erosion_rates_Point.png")
plt.savefig(out_png_rates, dpi=450)
plt.close()

print("Saved rates plot:", out_png_rates)


# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

# Panel 1: per-step mean Δz
axes[0].plot(times, dz_ts, marker='o', lw=1.8)
axes[0].axhline(0, color='k', linewidth=1, linestyle='--')
axes[0].set_ylabel("Mean Δz per step (m)\ninside buffer")
axes[0].set_title(f"Net Change in Elevation Through Time\nBuffered Region – {base}")
axes[0].grid(True, linestyle='--', alpha=0.5)

# Panel 2: cumulative net Δz
axes[1].plot(times, cum_dz_ts, marker='o', lw=1.8)
axes[1].axhline(0, color='k', linewidth=1, linestyle='--')
axes[1].set_ylabel("Cumulative mean Δz (m)\ninside buffer")
axes[1].set_xlabel("Time (yrs)")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_png = os.path.join(OUT_DIRpng, f"{base}_change_in_elevation_timeseries_BUFFERED.png")
plt.savefig(out_png, dpi=450)
plt.close()

print("Saved buffered change-in-elevation time series plot:", out_png)

#%%
import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import geometry_mask

# Reuse your globals:
# BASE_DIR, OUT_DIRtiff, OUT_DIRpng, INPUT_TIFF, BUFFER_SHP

TIFF_DIR = OUT_DIRtiff
base = os.path.splitext(INPUT_TIFF)[0]

# File patterns:
#   ext17_soil_produced_5000000yrs.tif
#   ext17_change_in_elevation_5000000yrs.tif
#   ext17_total_soil_depth_5000000yrs.tif
pattern_prod = re.compile(rf"{base}_production_rate_(\d+)yrs")
pattern_dz   = re.compile(rf"{base}_change_in_elevation_(\d+)yrs")
pattern_tot  = re.compile(rf"{base}_total_soil_depth_(\d+)yrs")


def build_point_buffer_mask(dem_path, point_shapefile, target_point_id, buffer_distance):
    """
    Create a True/False mask for ONLY the specified Point_ID.
    Returns a mask aligned with DEM orientation (after np.flipud).
    """
    import geopandas as gpd
    from rasterio.features import geometry_mask
    import rasterio
    import numpy as np

    # --- Load the DEM ---
    with rasterio.open(dem_path) as src:
        transform = src.transform
        out_shape = src.read(1).shape
        dem_crs = src.crs

    # --- Load points and select ONLY the one you want ---
    gdf = gpd.read_file(point_shapefile)

    field = "id"   # <--- change to whatever your field actually is

    if field not in gdf.columns:
        raise ValueError(f"Shapefile does not contain field: {field}")

    pt = gdf[gdf[field] == target_point_id]

    if pt.empty:
        raise ValueError(f"Point_ID {target_point_id} not found in shapefile.")

    # --- Reproject point to DEM CRS ---
    if pt.crs != dem_crs:
        pt = pt.to_crs(dem_crs)

    # --- Buffer only this point ---
    buf = pt.geometry.buffer(buffer_distance)

    # --- Rasterize mask (True inside buffer) ---
    mask_raw = geometry_mask(
        buf,
        transform=transform,
        invert=True,
        out_shape=out_shape
    )

    # Your TIFFs are saved with np.flipud, so flip mask too
    mask = np.flipud(mask_raw)

    return mask



# Build buffer mask once
dem_path = os.path.join(BASE_DIR, INPUT_TIFF)
target_point_id = 1   # <--- whichever point you want
buffer_mask = build_point_buffer_mask(
    dem_path=dem_path,
    point_shapefile=POINTS_SHP,
    target_point_id=target_point_id,
    buffer_distance=BUFFER_DISTANCE
)


def load_masked(path):
    """Load a TIFF, flip vertically, apply buffer mask, convert nodata to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

    arr = np.flipud(arr)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    arr_masked = np.where(buffer_mask, arr, np.nan)
    return arr_masked


# --- Collect series: time -> mean value inside buffer ---
mean_prod  = {}  # soil produced per step
mean_dz    = {}  # net change in elevation per step (Δz)
mean_depth = {}  # total soil depth

files = os.listdir(TIFF_DIR)

for fname in files:
    fpath = os.path.join(TIFF_DIR, fname)

    # Soil produced
    m_p = pattern_prod.search(fname)
    if m_p:
        t = int(m_p.group(1))
        arr = load_masked(fpath)
        mean_prod[t] = np.nanmean(arr)
        continue

    # Change in elevation
    m_z = pattern_dz.search(fname)
    if m_z:
        t = int(m_z.group(1))
        arr = load_masked(fpath)
        mean_dz[t] = np.nanmean(arr)  # negative = net erosion
        continue

    # Total soil depth
    m_t = pattern_tot.search(fname)
    if m_t:
        t = int(m_t.group(1))
        arr = load_masked(fpath)
        mean_depth[t] = np.nanmean(arr)
        continue

# --- Build unified time axis ---
times = sorted(set(mean_prod.keys()) | set(mean_dz.keys()) | set(mean_depth.keys()))

prod_ts  = [mean_prod.get(t, np.nan)  for t in times]
dz_ts    = [mean_dz.get(t, np.nan)    for t in times]
depth_ts = [mean_depth.get(t, np.nan) for t in times]

# Cumulative curves (depth-wise means)
cum_prod_ts = np.nancumsum(prod_ts)   # cumulative soil produced
cum_dz_ts   = np.nancumsum(dz_ts)     # cumulative net Δz (can go negative)

# --- Single plot with all three ---
plt.figure(figsize=(7, 5))
plt.gca().set_facecolor("#f0f0f0")   # light gray background
plt.xlim(0, 2500)
plt.ylim(0, 1.2)
plt.plot(times, cum_prod_ts, marker='o', linewidth=1.8, label="Cumulative soil produced (m)")
plt.plot(times, cum_dz_ts,   marker='o', linewidth=1.8, label="Cumulative net Δelevation (m)")
plt.plot(times, depth_ts,    marker='o', linewidth=1.8, label="Mean total soil depth (m)")

# plt.axhline(0, linestyle='--', linewidth=1.0)
# plt.yscale("log")   # <---- LOG SCALE HERE
#plt.xlabel("Time (yrs)")
#plt.ylabel("Depth / Elevation (m)\n(mean inside buffer)")
#plt.title(f"Soil Budget & Surface Change Through Time\nBuffered Region – {base}")
plt.grid(True, linestyle='--', alpha=0.5)
#plt.legend()
plt.tight_layout()

out_png = os.path.join(OUT_DIRpng, f"{base}_cum_prod_dz_totaldepth_BUFFERED.png")
plt.savefig(out_png, dpi=450)
plt.close()

print("Saved combined soil budget plot:", out_png)

#%%
#%%  COMBINED PLOT + TRANSPORT-ONLY INFILLING RATE (EXCLUDES PRODUCTION)

import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import geometry_mask

TIFF_DIR = OUT_DIRtiff
base = os.path.splitext(INPUT_TIFF)[0]

# File patterns:
pattern_prod = re.compile(rf"{base}_production_rate_(\d+)yrs")          # saved as increment (m) b/c you multiplied by dt
pattern_dz   = re.compile(rf"{base}_change_in_elevation_(\d+)yrs")      # Δz per step (m)
pattern_tot  = re.compile(rf"{base}_total_soil_depth_(\d+)yrs")         # total h (m)
pattern_dh   = re.compile(rf"{base}_change_in_soil_depth_(\d+)yrs")     # Δh per step (m)

def build_point_buffer_mask(dem_path, point_shapefile, target_point_id, buffer_distance):
    """
    Create a True/False mask for ONLY the specified point.
    Returns a mask aligned with arrays after np.flipud().
    """
    with rasterio.open(dem_path) as src:
        transform = src.transform
        out_shape = src.read(1).shape
        dem_crs = src.crs

    gdf = gpd.read_file(point_shapefile)

    field = "id"  # <-- CHANGE IF NEEDED
    if field not in gdf.columns:
        raise ValueError(f"Shapefile does not contain field: {field}. "
                         f"Available fields: {list(gdf.columns)}")

    pt = gdf[gdf[field] == target_point_id]
    if pt.empty:
        raise ValueError(f"{field}={target_point_id} not found in shapefile.")

    if pt.crs != dem_crs:
        pt = pt.to_crs(dem_crs)

    buf = pt.geometry.buffer(buffer_distance)

    mask_raw = geometry_mask(
        buf,
        transform=transform,
        invert=True,
        out_shape=out_shape
    )

    # TIFFs were saved with np.flipud(data), so flip mask to match loaded arrays
    return np.flipud(mask_raw)

# --- Build mask once ---
dem_path = os.path.join(BASE_DIR, INPUT_TIFF)
target_point_id = 1   # <-- choose your point id here
buffer_mask = build_point_buffer_mask(
    dem_path=dem_path,
    point_shapefile=POINTS_SHP,
    target_point_id=target_point_id,
    buffer_distance=BUFFER_DISTANCE
)

def load_masked(path):
    """Load a TIFF, flip vertically, apply buffer mask, convert nodata to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

    arr = np.flipud(arr)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return np.where(buffer_mask, arr, np.nan)

# --- Collect series: time -> mean value inside buffer ---
mean_prod  = {}  # Δh_prod per step (m) (your production_rate already includes *dt)
mean_dz    = {}  # Δz per step (m)
mean_depth = {}  # total soil depth h (m)
mean_dh    = {}  # Δh per step (m) (net soil thickness change)

for fname in os.listdir(TIFF_DIR):
    fpath = os.path.join(TIFF_DIR, fname)

    m = pattern_prod.search(fname)
    if m:
        t = int(m.group(1))
        mean_prod[t] = np.nanmean(load_masked(fpath))
        continue

    m = pattern_dz.search(fname)
    if m:
        t = int(m.group(1))
        mean_dz[t] = np.nanmean(load_masked(fpath))
        continue

    m = pattern_tot.search(fname)
    if m:
        t = int(m.group(1))
        mean_depth[t] = np.nanmean(load_masked(fpath))
        continue

    m = pattern_dh.search(fname)
    if m:
        t = int(m.group(1))
        mean_dh[t] = np.nanmean(load_masked(fpath))
        continue

# --- Unified time axis (only keep times where we have what's needed) ---
times = sorted(set(mean_prod.keys()) | set(mean_dz.keys()) | set(mean_depth.keys()) | set(mean_dh.keys()))

prod_ts  = np.array([mean_prod.get(t, np.nan)  for t in times], dtype=float)  # m per step
dz_ts    = np.array([mean_dz.get(t, np.nan)    for t in times], dtype=float)  # m per step
depth_ts = np.array([mean_depth.get(t, np.nan) for t in times], dtype=float)  # m
dh_ts    = np.array([mean_dh.get(t, np.nan)    for t in times], dtype=float)  # m per step

# --- Cumulative curves (depth-wise means; units = m) ---
cum_prod_ts = np.nancumsum(prod_ts)
cum_dz_ts   = np.nancumsum(dz_ts)

# --- TRANSPORT-ONLY: exclude production ---
# Δh_transport per step (m) = Δh - Δh_prod
transport_dh_ts = dh_ts - prod_ts

# Net transport rate (m/yr): can be +/- (infill or evacuation)
dt_model = dt  # dt from your simulation (50 yrs)
transport_net_rate_ts = transport_dh_ts / dt_model

# Transport-only INFILLING rate (m/yr): deposition only (positive)
transport_infill_rate_ts = np.maximum(0.0, transport_dh_ts) / dt_model

# Optional: cumulative transport-only infill (m) (deposition only)
cum_transport_infill_ts = np.nancumsum(np.maximum(0.0, transport_dh_ts))

# --- Start curves at t=0 with 0 ---
times_arr = np.array(times, dtype=float)

times_plot      = np.insert(times_arr, 0, 0.0)
cum_prod_plot   = np.insert(cum_prod_ts, 0, 0.0)
cum_dz_plot     = np.insert(cum_dz_ts,   0, 0.0)
depth_plot      = np.insert(depth_ts,    0, 0.0)  # you said hollow starts at 0

# For rate plots, it’s also nice to start at 0
transport_net_rate_plot    = np.insert(transport_net_rate_ts,    0, 0.0)
transport_infill_rate_plot = np.insert(transport_infill_rate_ts, 0, 0.0)
cum_transport_infill_plot  = np.insert(cum_transport_infill_ts,  0, 0.0)

# -------------------------
# Plot 1: Your original combined plot (cumulative prod, cumulative dz, total depth)
# -------------------------
plt.figure(figsize=(7, 5))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")

ax.set_xlim(0, 2500)
ax.set_ylim(0, 1.2)

ax.plot(times_plot, cum_prod_plot, marker='o', lw=1.8, label="Cumulative soil produced (m)")
ax.plot(times_plot, cum_dz_plot,   marker='o', lw=1.8, label="Cumulative net Δelevation (m)")
ax.plot(times_plot, depth_plot,    marker='o', lw=1.8, label="Mean total soil depth (m)")

ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

out_png = os.path.join(OUT_DIRpng, f"{base}_cum_prod_dz_totaldepth_Point{target_point_id}.png")
plt.savefig(out_png, dpi=450)
plt.close()
print("Saved combined soil budget plot:", out_png)

# -------------------------
# Plot 2: Transport-only infilling rate (m/yr) (what you asked for)
# -------------------------
plt.figure(figsize=(7, 4.5))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")

ax.set_xlim(0, 2500)

ax.plot(times_plot, transport_infill_rate_plot, marker='o', lw=1.8,
        label="Transport-only infilling rate (m/yr)")

ax.axhline(0, color='k', lw=1, ls='--')
ax.set_xlabel("Time (yrs)")
ax.set_ylabel("Rate (m/yr)\n(mean inside hollow buffer)")
ax.set_title(f"Transport-only Infilling Rate (excludes production)\nPoint {target_point_id} – {base}")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

out_png2 = os.path.join(OUT_DIRpng, f"{base}_transport_only_infilling_rate_Point{target_point_id}.png")
plt.savefig(out_png2, dpi=450)
plt.close()
print("Saved transport-only infilling rate plot:", out_png2)

# -------------------------
# Optional Plot 3: net transport rate (can be +/-) + cumulative transport-only infill
# -------------------------
plt.figure(figsize=(7, 4.5))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")
ax.set_xlim(0, 2500)

ax.plot(times_plot, transport_net_rate_plot, marker='o', lw=1.8,
        label="Net transport rate (m/yr; +/-)")
ax.axhline(0, color='k', lw=1, ls='--')

ax.set_xlabel("Time (yrs)")
ax.set_ylabel("Rate (m/yr)\n(mean inside hollow buffer)")
ax.set_title(f"Net Transport Rate (excludes production)\nPoint {target_point_id} – {base}")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

out_png3 = os.path.join(OUT_DIRpng, f"{base}_transport_net_rate_Point{target_point_id}.png")
plt.savefig(out_png3, dpi=450)
plt.close()
print("Saved net transport rate plot:", out_png3)

plt.figure(figsize=(7, 4.5))
ax = plt.gca()
ax.set_facecolor("#f0f0f0")
ax.set_xlim(0, 2500)

ax.plot(times_plot, cum_transport_infill_plot, marker='o', lw=1.8,
        label="Cumulative transport-only infill (m)")
ax.set_xlabel("Time (yrs)")
ax.set_ylabel("Cumulative infill (m)\n(mean inside hollow buffer)")
ax.set_title(f"Cumulative Transport-only Infilling (excludes production)\nPoint {target_point_id} – {base}")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

out_png4 = os.path.join(OUT_DIRpng, f"{base}_cum_transport_only_infill_Point{target_point_id}.png")
plt.savefig(out_png4, dpi=450)
plt.close()
print("Saved cumulative transport-only infill plot:", out_png4)
