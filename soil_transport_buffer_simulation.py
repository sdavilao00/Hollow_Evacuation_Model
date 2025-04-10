# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:32:06 2024

@author: Selina Davila Olivera
Updated by ChatGPT on Apr 3, 2025
"""

# All required imports
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

# Setup paths and global constants
BASE_DIR = os.path.join(os.getcwd(), 'ExampleDEM')
INPUT_TIFF = 'mdstab_smooth_nowrap.tif'

OUT_DIR = os.path.join(BASE_DIR, 'simulation_results')
OUT_DIRpng = os.path.join(OUT_DIR, 'PNGs')
OUT_DIRtiff = os.path.join(OUT_DIR, 'GeoTIFFs')
OUT_DIRasc = os.path.join(OUT_DIR, 'ASCs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIRpng, exist_ok=True)
os.makedirs(OUT_DIRtiff, exist_ok=True)
os.makedirs(OUT_DIRasc, exist_ok=True)

ft2mUS = 1200 / 3937
ft2mInt = 0.3048
BUFFER_DISTANCE = 6

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
    if gdf.crs is not None:
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
            if gdf.crs != dem_crs:
                print(f"Reprojecting shapefile from {gdf.crs} to {dem_crs}")
                gdf = gdf.to_crs(dem_crs)
            transform = src.transform
    else:
        print("Warning: Shapefile has no CRS. Assuming it's aligned with the DEM.")
        with rasterio.open(dem_path) as src:
            transform = src.transform

    buffered_geoms = gdf.buffer(buffer_distance)
    mask = geometry_mask([buffered_geoms.unary_union], transform=transform, invert=True, out_shape=grid.shape)

    if 'soil__depth' not in grid.at_node:
        soil_depth = np.full(grid.number_of_nodes, 0.5)
        grid.add_field('soil__depth', soil_depth, at='node')
    else:
        soil_depth = grid.at_node['soil__depth']
    soil_depth[mask.flatten()] = 0
    return grid

def init_simulation(asc_file, K, Sc, XYZunit=None, shapefile=None, buffer_distance=BUFFER_DISTANCE, dem_path=None):
    grid, _ = read_esri_ascii(asc_file, name='topographic__elevation')
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)
    if shapefile and dem_path:
        grid = apply_buffer_to_soil_depth(grid, shapefile, buffer_distance, dem_path)
    else:
        soil_depth = np.full(grid.number_of_nodes, 0.5)
        grid.add_field('soil__depth', soil_depth, at='node')
    if XYZunit is None or 'meter' in XYZunit.lower():
        Kc = K
    elif 'foot' in XYZunit.lower():
        Kc = K / (ft2mUS ** 2) if "US" in XYZunit else K / (ft2mInt ** 2)
    else:
        raise RuntimeError("Unsupported unit type for K conversion.")
    TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=Kc, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable="pass")
    return grid, TNLD

# Additional plotting and save functions skipped for brevity...
# Assume the rest of your previous code continues below this point...
