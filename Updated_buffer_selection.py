# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:27:22 2025

@author: sdavilao
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
import numpy as np
from osgeo import gdal
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

#%% ---------------- LOAD & PROCESS DATA ----------------
# Define file paths
shapefile_path = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Desktop\QGIS\HC\04282023\MDSTAB_hollow.shp"
dem_path = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Desktop\QGIS\HC\04282023\avg9.tif"
slope_path = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Desktop\QGIS\HC\04282023\slope.tif"

# Define soil depth raster directory and naming pattern
soil_depth_dir = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Documents\3D_test\ExampleDEM\simulation_results\GeoTIFFS\reprojected"
basename = "MDSTAB_Test"
soil_depth_pattern = os.path.join(soil_depth_dir, f"{basename}_total_soil_depth_*yrs.tif")

# Load the point shapefile
gdf = gpd.read_file(shapefile_path)

# Ensure the shapefile and raster CRS match
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs  
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)  

# Get all soil depth .tif files and sort them by year
soil_depth_files = sorted(glob.glob(soil_depth_pattern))

# Define buffer sizes to test
buffer_sizes = [2, 4, 6, 8]

# Create a list to store results
results = []

# Loop through each point in the shapefile
for idx, point_geom in enumerate(gdf.geometry):
    for buffer_distance in buffer_sizes:
        buffer_geom = point_geom.buffer(buffer_distance)
        buffer_json = [buffer_geom.__geo_interface__]

        ### Step 1: Compute Slope within the Buffer ###
        try:
            with rasterio.open(slope_path) as slope_raster:
                slope_image, _ = mask(slope_raster, buffer_json, crop=True)
                slope_image = slope_image[0]  
                valid_slope_values = slope_image[slope_image != slope_raster.nodata]
                avg_slope = np.mean(valid_slope_values) if valid_slope_values.size > 0 else np.nan
        except Exception:
            avg_slope = np.nan  

        ### Step 2: Extract Soil Depth for Each Year ###
        for soil_depth_tif in soil_depth_files:
            match = re.search(r'_(\d+)yrs\.tif$', soil_depth_tif)
            if match:
                year = int(match.group(1))  
            else:
                continue  

            with rasterio.open(soil_depth_tif) as soil_depth_raster:
                soil_depth_image, _ = mask(soil_depth_raster, buffer_json, crop=True)
                soil_depth_image = soil_depth_image[0]  
                valid_soil_depth_values = soil_depth_image[soil_depth_image != soil_depth_raster.nodata]
                avg_soil_depth = np.mean(valid_soil_depth_values) if valid_soil_depth_values.size > 0 else np.nan

            # Store results in a dictionary
            results.append({
                'Point_ID': idx,
                'Year': year,
                'Buffer_Size': buffer_distance,
                'Avg_Slope': avg_slope,
                'Avg_Soil_Depth': avg_soil_depth
            })

# Convert results to DataFrame
df = pd.DataFrame(results)

# Sort DataFrame
df = df.sort_values(by=['Point_ID', 'Year', 'Buffer_Size']).reset_index(drop=True)

#%% ---------------- CALCULATE FACTOR OF SAFETY ----------------
# Constants
Klin = 0.004  
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
C0 = 6400  
j = 1.5  

def calculate_fs(row):
    hollow_rad = np.radians(row['Avg_Slope'])
    z = row['Avg_Soil_Depth']
    
    if np.isnan(z) or np.isnan(hollow_rad):
        return np.nan  
    
    Crb = C0 * np.exp(-z * j)
    Crl = (C0 / (j * z)) * (1 - np.exp(-z * j))

    K0 = 1 - np.sin(hollow_rad)

    aa = 8 * (Crl / (ys * z)) * (np.cos(hollow_rad) ** 2) * np.sin(phi) * np.cos(phi)
    bb = 4 * (Crl / (ys * z)) * (np.cos(phi) ** 2)
    cc = 4 * (np.cos(hollow_rad) ** 2) * (((np.cos(hollow_rad) ** 2) - (np.cos(phi) ** 2)))
    dd = 2 * ((Crl / (ys * z))) * (np.cos(phi) * np.sin(phi))
    ee = 2 * (np.cos(hollow_rad)) ** 2
    ff = 1 / ((np.cos(phi) ** 2))

    Kp = ff * ((ee) + (dd) + ((cc) + (bb) + (aa)) ** 0.5) - 1
    Ka = ff * ((ee) + (dd) - ((cc) + (bb) + (aa)) ** 0.5) - 1

    Frb = (Crb + ((np.cos(hollow_rad)) ** 2) * z * (ys - yw * m) * np.tan(phi)) * l * w
    Frc = (Crl + (K0 * 0.5 * z * (ys - yw * m ** 2) * np.tan(phi))) * (np.cos(hollow_rad) * z * l * 2)
    Frddu = (Kp - Ka) * 0.5 * (z ** 2) * (ys - yw * (m ** 2)) * w
    Fdc = (np.sin(hollow_rad)) * (np.cos(hollow_rad)) * z * ys * l * w

    return (Frb + Frc + Frddu) / Fdc if Fdc != 0 else np.nan

df['FS'] = df.apply(calculate_fs, axis=1)

#%% ---------------- DETERMINE OPTIMAL BUFFER SIZE & SAVE EXTRA DATA ----------------
optimal_buffers = {}

for point_id in df['Point_ID'].unique():
    point_data = df[df['Point_ID'] == point_id]

    # Step 1: Find the earliest year where FS is closest to 1 for any buffer size
    valid_data = point_data.copy()
    valid_data['FS_Diff'] = abs(valid_data['FS'] - 1)  # Compute FS difference from 1

    # Sort by Year (earliest first) and then by FS closest to 1
    sorted_data = valid_data.sort_values(by=['Year', 'FS_Diff'])

    if sorted_data.empty:
        print(f"Warning: No valid FS data for Point_ID {point_id}. Skipping.")
        continue

    best_buffer_row = sorted_data.iloc[0]  # Selects the earliest FS ≈ 1 occurrence
    optimal_buffer = best_buffer_row['Buffer_Size']  # The best buffer

    # Step 2: Within that optimal buffer, find the year where FS is closest to 1
    buffer_data = point_data[point_data['Buffer_Size'] == optimal_buffer].copy()
    buffer_data['FS_Diff'] = abs(buffer_data['FS'] - 1)  # Recalculate FS diff

    best_year_row = buffer_data.loc[buffer_data['FS_Diff'].idxmin()]  # Select year closest to FS = 1

    # Step 3: Extract the corresponding values
    optimal_buffers[point_id] = {
        'Optimal_Buffer': optimal_buffer,  # Best buffer size
        'Year': best_year_row['Year'],  # Year closest to FS = 1 within that buffer
        'FS': best_year_row['FS'],  # Actual FS value
        'Avg_Soil_Depth': best_year_row['Avg_Soil_Depth'],  # Corresponding soil depth
        'Avg_Slope': best_year_row['Avg_Slope']  # Corresponding slope
    }

# Convert to DataFrame
df_optimal = pd.DataFrame.from_dict(optimal_buffers, orient='index')
df_optimal.reset_index(inplace=True)
df_optimal.rename(columns={'index': 'Point_ID'}, inplace=True)

# Save to CSV file
output_path = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Desktop\QGIS\HC\04282023\optimal_buffer_results.csv"
df_optimal.to_csv(output_path, index=False)

# Print Results
print(df_optimal)










