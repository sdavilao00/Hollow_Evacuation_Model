# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:41:44 2025

@author: sdavilao
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
import numpy as np
from osgeo import gdal

# Define file paths
shapefile_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/MDSTAB_test.shp"  # Shapefile containing the point
dem_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/avg9.tif"  # DEM raster
soil_depth_tif = "C:/Users\sdavilao/OneDrive - University Of Oregon/Documents/3D_test/warp_soildepth_mdtst.tif"  # Soil depth raster
slope_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/slope.tif"

# Load the point shapefile
gdf = gpd.read_file(shapefile_path)

# Extract the first point (modify if multiple points exist)
point_geom = gdf.geometry.iloc[0]  # Assuming a single point

# Create a buffer around the point (e.g., 100m radius)
buffer_distance = 5  # Set buffer distance in meters
buffer_geom = point_geom.buffer(buffer_distance)

# Ensure the shapefile and raster CRS match
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs  # Get CRS from DEM
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)  # Reproject vector data

# Convert buffer to GeoJSON-like format
buffer_json = [buffer_geom.__geo_interface__]

### STEP 1: PROCESS DEM SLOPE ###
# Compute slope if not already computed


try:
    with rasterio.open(slope_path) as slope_raster:
        slope_image, _ = mask(slope_raster, buffer_json, crop=True)
        slope_image = slope_image[0]  # Extract slope band
        valid_slope_values = slope_image[slope_image != slope_raster.nodata]
        avg_slope = np.mean(valid_slope_values) if valid_slope_values.size > 0 else np.nan
except Exception:
    print("Slope raster not found. Computing slope from DEM...")
    slope_output_path = "temp_slope.tif"
    gdal.DEMProcessing(slope_output_path, dem_path, "slope", computeEdges=True)
    
    with rasterio.open(slope_output_path) as slope_raster:
        slope_image, _ = mask(slope_raster, buffer_json, crop=True)
        slope_image = slope_image[0]  # Extract slope band
        valid_slope_values = slope_image[slope_image != slope_raster.nodata]
        avg_slope = np.mean(valid_slope_values) if valid_slope_values.size > 0 else np.nan

### STEP 2: PROCESS SOIL DEPTH ###
with rasterio.open(soil_depth_tif) as soil_depth_raster:
    soil_depth_image, _ = mask(soil_depth_raster, buffer_json, crop=True)
    soil_depth_image = soil_depth_image[0]  # Extract depth band
    valid_soil_depth_values = soil_depth_image[soil_depth_image != soil_depth_raster.nodata]
    avg_soil_depth = np.mean(valid_soil_depth_values) if valid_soil_depth_values.size > 0 else np.nan

# Print results
print(f"Average Slope within Buffer: {avg_slope:.2f} degrees")
print(f"Average Soil Depth within Buffer: {avg_soil_depth:.2f} meters")
