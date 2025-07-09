# -*- coding: utf-8 -*-
"""
Complete soil transport simulation with CRS-aware buffer masking and plotting.
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import Point
from rasterio.features import geometry_mask
from osgeo import gdal
from landlab import imshowhs_grid
from landlab.components import TaylorNonLinearDiffuser
from landlab.io import read_esri_ascii, write_esri_ascii

# Paths and constants
BASE_DIR = os.path.join(os.getcwd())

BUFFER_DISTANCE = 7

ft2mUS = 1200 / 3937
ft2mInt = 0.3048

OUT_DIR = os.path.join(BASE_DIR, 'simulation_results')
OUT_DIRpng = os.path.join(OUT_DIR, 'PNGs')
OUT_DIRtiff = os.path.join(OUT_DIR, 'GeoTIFFs')
OUT_DIRasc = os.path.join(OUT_DIR, 'ASCs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIRpng, exist_ok=True)
os.makedirs(OUT_DIRtiff, exist_ok=True)
os.makedirs(OUT_DIRasc, exist_ok=True)

#%%
def create_buffer_from_points(input_points_path, output_buffer_path, buffer_distance, target_crs=None):
    gdf = gpd.read_file(input_points_path)

    if target_crs:
        if gdf.crs is None:
            print("⚠️  Shapefile has no CRS. Assigning and reprojecting to DEM CRS.")
            gdf = gdf.set_crs(target_crs, allow_override=True)
            gdf = gdf.to_crs(target_crs)
        elif gdf.crs != target_crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {target_crs}")
            gdf = gdf.to_crs(target_crs)
    else:
        print("⚠️ No target CRS provided. Assuming points are already aligned with DEM.")

    buffer_geom = gdf.buffer(buffer_distance)
    buffered = gdf.copy()
    buffered["geometry"] = buffer_geom
    buffered.set_crs(gdf.crs, inplace=True)
    buffered.to_file(output_buffer_path)
    print(f"✅ Buffer created at {output_buffer_path}")
    print("Buffer bounds:", buffered.total_bounds)
    return output_buffer_path
#%%
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
#%%
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
    print(f"Applied 0m soil depth to buffer zones. Total zero-depth cells: {np.sum(soil_depth == 0.01)}")
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


#%%
def init_simulation(asc_file, K, Sc, XYZunit=None, shapefile=None, buffer_distance=BUFFER_DISTANCE, dem_path=None):
    grid, _ = read_esri_ascii(asc_file, name='topographic__elevation')
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    # Initialize with 0.5m soil depth
    soil_depth = np.full(grid.number_of_nodes, 0.5)
    grid.add_field('soil__depth', soil_depth, at='node')

    if shapefile and dem_path:
        grid = apply_buffer_to_soil_depth(grid, shapefile, buffer_distance, dem_path)

    if XYZunit is None or 'meter' in XYZunit.lower():
        Kc = K
    elif 'foot' in XYZunit.lower():
        Kc = K / (ft2mUS ** 2) if "US" in XYZunit else K / (ft2mInt ** 2)
    else:
        raise RuntimeError("Unsupported unit type for K conversion.")

    TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=Kc, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable="pass")
    return grid, TNLD
#%%
def plot_change(data, title, basefilename, time, K, grid_shape,
                vmin=None, vmax=None, cmap=None, output_dir=None):
    if data.ndim == 1:
        data = data.reshape(grid_shape)
    plt.figure(figsize=(6, 5.25))
    plt.imshow(np.flipud(data), cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Value')
    plt.title(f"{title} at {time} yrs (K = {K})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    # Save to passed-in output directory
    png_path = os.path.join(output_dir, f"{basefilename}_{title.replace(' ', '_')}_{time}yrs_K{K}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

#%%
def save_as_tiff(data, filename, meta, grid_shape):
    if data.ndim == 1:
        data = data.reshape(grid_shape)
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(np.flipud(data.astype(rasterio.float32)), 1)
    print(f"Saved TIFF: {filename}")
#%%
def plot_save(grid, z, basefilename, time, K, mean_res, XYZunit, output_dir_png, output_dir_asc):
    plt.figure(figsize=(6, 5.25))
    imshowhs_grid(grid, z, plot_type="Hillshade")
    plt.title(f"{basefilename} Time {time} yrs (K = {K})")
    plt.tight_layout()
    png_path = os.path.join(output_dir_png, f"{basefilename}_{time}yrs_K{K}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    asc_path = os.path.join(output_dir_asc, f"{basefilename}_{time}yrs_K{K}.asc")
    write_esri_ascii(asc_path, grid, names=['topographic__elevation'], clobber=True)
    return asc_path

#%%
def run_simulation(in_tiff, K, Sc, dt, target_time, point_shapefile,
                   OUT_DIRpng, OUT_DIRtiff, OUT_DIRasc):
    dem_path = os.path.join(BASE_DIR, in_tiff)

    # 🔁 Open the DEM and use its CRS when creating the buffer
    with rasterio.open(dem_path) as src:
        buffer_name = os.path.splitext(os.path.basename(point_shapefile))[0] + "_buffer.shp"
        buffer_path = os.path.join(BASE_DIR, buffer_name)
        buffer_shapefile = create_buffer_from_points(point_shapefile, buffer_path, BUFFER_DISTANCE, src.crs)

    basefilename = os.path.splitext(in_tiff)[0]
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(dem_path, in_asc)

    grid, tnld = init_simulation(in_asc, K, Sc, XYZunit, shapefile=buffer_shapefile, dem_path=dem_path)
    z_old = grid.at_node['topographic__elevation']
    soil_depth = grid.at_node['soil__depth']
    initial_soil_depth = soil_depth.copy()
    total_soil_depth = soil_depth.copy()

    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()

    asc_path = plot_save(grid, z_old, basefilename, 0, K, mean_res, XYZunit, OUT_DIRpng, OUT_DIRasc)

    num_steps = int(target_time / dt)
    for i in range(num_steps):
        tnld.run_one_step(dt)
        time = (i + 1) * dt

        production_rate = (pr / ps) * (P0 * np.exp(-total_soil_depth / h0)) * dt
        z_new = grid.at_node['topographic__elevation']
        elevation_change = z_new - z_old

        nonzero_soil_mask = initial_soil_depth > 0.00
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
            plot_change(elevation_change, "Change in Elevation", basefilename, time, K, grid_shape, vmin=-1, vmax=1, cmap='coolwarm', output_dir=OUT_DIRpng)
            plot_change(change_in_soil_depth, "Change in Soil Depth", basefilename, time, K, grid_shape, vmin=-1, vmax=1, cmap='RdYlBu', output_dir=OUT_DIRpng)
            plot_change(total_soil_depth, "Total Soil Depth", basefilename, time, K, grid_shape, vmin=0, vmax=5, cmap='viridis', output_dir=OUT_DIRpng)
            plot_change(production_rate, "Soil Produced", basefilename, time, K, grid_shape, vmin=0, vmax=0.01, cmap='plasma', output_dir=OUT_DIRpng)
        
        if time % 50 == 0:
            asc_path = plot_save(grid, z_new, basefilename, time, K, mean_res, XYZunit, OUT_DIRpng, OUT_DIRasc)
            tiff_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs_K{K}.tif")
            asc_to_tiff(asc_path, tiff_path, meta)

    os.remove(in_asc)
    os.remove(in_asc.replace(".asc", ".prj"))
    print("Simulation complete.")

# %% Batch mode: Run simulations over multiple DEM + SHP files
import glob

# Example run parameters
K = 0.0042
Sc = 1.25
pr = 2000
ps = 1000
P0 = 0.0003
h0 = 0.5
dt = 50
target_time = 150

BUFFER_DISTANCE = 8

# Loop through each DEM
dem_files = sorted(glob.glob(os.path.join(BASE_DIR, "*.tif")))

for dem_path in dem_files:
    dem_name = os.path.splitext(os.path.basename(dem_path))[0]
    shp_path = os.path.join(BASE_DIR, f"{dem_name}.shp")

    if not os.path.exists(shp_path):
        print(f"❌ No shapefile for {dem_name}. Skipping...")
        continue

    # Create per-DEM output folders
    DEM_OUT_DIR = os.path.join(OUT_DIR, dem_name)
    DEM_OUT_PNG = os.path.join(DEM_OUT_DIR, 'PNGs')
    DEM_OUT_TIFF = os.path.join(DEM_OUT_DIR, 'GeoTIFFs')
    DEM_OUT_ASC = os.path.join(DEM_OUT_DIR, 'ASCs')
    for folder in [DEM_OUT_DIR, DEM_OUT_PNG, DEM_OUT_TIFF, DEM_OUT_ASC]:
        os.makedirs(folder, exist_ok=True)

    print(f"\n🚀 Running simulation for {dem_name}")
    try:
        run_simulation(
            in_tiff=os.path.basename(dem_path),
            K=K,
            Sc=Sc,
            dt=dt,
            target_time=target_time,
            point_shapefile=shp_path,
            OUT_DIRpng=DEM_OUT_PNG,
            OUT_DIRtiff=DEM_OUT_TIFF,
            OUT_DIRasc=DEM_OUT_ASC
        )
    except Exception as e:
        print(f"❗ Simulation failed for {dem_name}: {e}")
