# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:27:07 2025

@author: sdavilao
"""

import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
import numpy as np
from osgeo import gdal
import re


# Define the target CRS (change EPSG code as needed)
target_crs = "EPSG:32610"  # UTM Zone 10N, change if needed

# Directory containing the soil depth TIFFs
soil_depth_dir = "C:/Users/sdavilao/OneDrive - University Of Oregon/Documents/3D_test/ExampleDEM/simulation_results/GeoTIFFs"
output_dir = os.path.join(soil_depth_dir, "reprojected")  # Create a folder for reprojected TIFFs
os.makedirs(output_dir, exist_ok=True)

# Loop through all TIFF files and reproject
for filename in os.listdir(soil_depth_dir):
    if filename.endswith(".tif"):
        input_tif = os.path.join(soil_depth_dir, filename)
        output_tif = os.path.join(output_dir, f"reprojected_{filename}")

        # Use GDAL Warp to reproject
        gdal.Warp(output_tif, input_tif, dstSRS=target_crs, resampleAlg="bilinear")

        print(f"Reprojected: {filename} → {output_tif}")

print("All TIFFs have been reprojected!")
#%%
# Define file paths
shapefile_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/MDSTAB_hollow.shp"  
dem_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/avg9.tif"  
slope_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/slope.tif"

# Directory containing time-stepped soil depth rasters
soil_depth_dir = "C:/Users/sdavilao\OneDrive - University Of Oregon/Documents/3D_test/ExampleDEM/simulation_results/GeoTIFFs"

# Load the point shapefile
gdf = gpd.read_file(shapefile_path)

# Extract the first point (modify if multiple points exist)
point_geom = gdf.geometry.iloc[0]  # Assuming a single point

# Create a buffer around the point (e.g., 5m radius)
buffer_distance = 2  # Set buffer distance in meters
buffer_geom = point_geom.buffer(buffer_distance)

# Ensure the shapefile and raster CRS match
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs  # Get CRS from DEM
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)  # Reproject vector data

# Convert buffer to GeoJSON-like format
buffer_json = [buffer_geom.__geo_interface__]

### STEP 1: PROCESS DEM SLOPE ###
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

### STEP 2: PROCESS SOIL DEPTH AT MULTIPLE TIME STEPS ###
# Find all potential soil depth files
all_files = os.listdir(soil_depth_dir)
soil_depth_files = [f for f in all_files if "_total_soil_depth_" in f and f.endswith(".tif")]

if not soil_depth_files:
    print("No soil depth raster files found.")
else:
    # Extract the generic basename dynamically
    match = re.match(r"(.+)_total_soil_depth_\d+yrs\.tif", soil_depth_files[0])
    if match:
        base_name = match.group(1)  # Extracts "MDSTAB_Test" dynamically
        print(f"Detected base name: {base_name}")
    else:
        print("Error: Could not detect a base name pattern.")
        base_name = "UnknownDataset"

    # Sort files by time step
    soil_depth_files = sorted(
        soil_depth_files,
        key=lambda x: int(re.search(r'(\d+)yrs', x).group(1))  # Sort by extracted time step
    )

    print(f"Processing {len(soil_depth_files)} time-stepped soil depth rasters...")

    # Iterate through each time-stepped soil depth raster
    for soil_depth_file in soil_depth_files:
        time_step = int(re.search(r'(\d+)yrs', soil_depth_file).group(1))  # Extract time step
        soil_depth_tif = os.path.join(soil_depth_dir, soil_depth_file)

        with rasterio.open(soil_depth_tif) as soil_depth_raster:
            soil_depth_image, _ = mask(soil_depth_raster, buffer_json, crop=True)
            soil_depth_image = soil_depth_image[0]  # Extract depth band
            valid_soil_depth_values = soil_depth_image[soil_depth_image != soil_depth_raster.nodata]
            avg_soil_depth = np.mean(valid_soil_depth_values) if valid_soil_depth_values.size > 0 else np.nan

        # Print results for each time step
        print(f"Time Step: {time_step} years")
        print(f" - Average Soil Depth: {avg_soil_depth:.2f} meters")
        
#%%
import matplotlib.pyplot as plt
import rasterio

# Open the soil depth raster
with rasterio.open(soil_depth_tif) as soil_raster:
    raster_bounds = soil_raster.bounds

    # Plot the raster extent
    plt.plot(
        [raster_bounds.left, raster_bounds.right, raster_bounds.right, raster_bounds.left, raster_bounds.left],
        [raster_bounds.bottom, raster_bounds.bottom, raster_bounds.top, raster_bounds.top, raster_bounds.bottom],
        label="Raster Bounds", color="red"
    )

# Ensure the buffer geometry is in the correct CRS
if gdf.crs != soil_raster.crs:
    gdf = gdf.to_crs(soil_raster.crs)
    buffer_geom = gdf.geometry.iloc[0].buffer(buffer_distance)  # Recompute buffer

# Plot the buffered point
x, y = buffer_geom.exterior.xy  # Get buffer coordinates
plt.plot(x, y, label="Buffered Point", color="blue")

# Formatting
plt.legend()
plt.xlabel("Longitude" if gdf.crs.is_geographic else "X Coordinate")
plt.ylabel("Latitude" if gdf.crs.is_geographic else "Y Coordinate")
plt.title("Raster Extent and Buffered Point")
plt.grid(True)

# Show the plot
plt.show()

#%%
with rasterio.open(soil_depth_tif) as soil_raster:
    raster_crs = soil_raster.crs
    vector_crs = gdf.crs

    print(f"Raster CRS: {raster_crs}")
    print(f"Vector CRS: {vector_crs}")

    # Ensure CRS match
    if vector_crs != raster_crs:
        print("⚠️ CRS Mismatch! Reprojecting shapefile to match raster...")
        gdf = gdf.to_crs(raster_crs)
        buffer_geom = gdf.geometry.iloc[0].buffer(buffer_distance)  # Recompute buffer
        buffer_json = [buffer_geom.__geo_interface__]
    else:
        print("✅ CRS Match! No need for reprojection.")
        
#%%
import rasterio
from shapely.geometry import box

with rasterio.open(soil_depth_tif) as soil_raster:
    raster_bounds = soil_raster.bounds
    raster_box = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)

    print("\n📏 Raster Bounds:")
    print(f"  Left: {raster_bounds.left}, Right: {raster_bounds.right}")
    print(f"  Bottom: {raster_bounds.bottom}, Top: {raster_bounds.top}")

# Check buffered point bounds
print("\n📍 Buffered Zone Bounds:")
print(f"  Left: {buffer_geom.bounds[0]}, Right: {buffer_geom.bounds[2]}")
print(f"  Bottom: {buffer_geom.bounds[1]}, Top: {buffer_geom.bounds[3]}")

# Check if buffer intersects the raster
if buffer_geom.intersects(raster_box):
    print("\n✅ The buffered zone INTERSECTS the raster.")
else:
    print("\n⚠️ Warning: The buffered zone DOES NOT intersect the raster. Check CRS, buffer size, or location.")

