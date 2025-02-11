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
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%

# Define file paths
shapefile_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/MDSTAB_hollow.shp"
dem_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/avg9.tif"
slope_path = "C:/Users/sdavilao/OneDrive - University Of Oregon/Desktop/QGIS/HC/04282023/slope.tif"

# Define soil depth raster directory and naming pattern
soil_depth_dir = "C:/Users/sdavilao/OneDrive - University Of Oregon/Documents/3D_test/ExampleDEM/simulation_results/GeoTIFFS/reprojected/"
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
        # Create a buffer around the point
        buffer_geom = point_geom.buffer(buffer_distance)
        buffer_json = [buffer_geom.__geo_interface__]

        ### Step 1: Compute Slope within the Buffer ###
        try:
            with rasterio.open(slope_path) as slope_raster:
                slope_image, _ = mask(slope_raster, buffer_json, crop=True)
                slope_image = slope_image[0]  # Extract slope band
                valid_slope_values = slope_image[slope_image != slope_raster.nodata]
                avg_slope = np.mean(valid_slope_values) if valid_slope_values.size > 0 else np.nan
        except Exception:
            print(f"Slope raster not found for Point {idx} and Buffer {buffer_distance}m.")

        ### Step 2: Extract Soil Depth for Each Year ###
        for soil_depth_tif in soil_depth_files:
            # Extract year from filename using regex
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
                'Avg_Slope': avg_slope,  # Now slope is calculated with the buffer
                'Avg_Soil_Depth': avg_soil_depth
            })

# Convert results to Pandas DataFrame
df = pd.DataFrame(results)

# Sort DataFrame by 'Point_ID', 'Year', and 'Buffer_Size'
df = df.sort_values(by=['Point_ID', 'Year', 'Buffer_Size']).reset_index(drop=True)

# Print DataFrame for debugging
print(df)

plt.figure(figsize=(10, 6))

# Loop through each buffer size and plot depth vs. time
for buffer_size in df['Buffer_Size'].unique():
    subset = df[df['Buffer_Size'] == buffer_size]  # Filter data for buffer size
    avg_depths = subset.groupby("Year")["Avg_Soil_Depth"].mean()  # Get avg depth per year

    plt.plot(avg_depths.index, avg_depths.values, marker='o', linestyle='-', label=f'Buffer {buffer_size}m')

# Labels and title
plt.xlabel("Year")
plt.ylabel("Average Soil Depth (meters)")
plt.title("Soil Depth Over Time for Different Buffer Sizes")
plt.legend(title="Buffer Size (m)")
plt.grid(True, linestyle="--", linewidth=0.5)

#%%
# Knon variables
Klin = 0.004 # m2/yr # linear sediment transport coefficient
Sc = 1.25 # unitless # critical slope

# Critical Depth variables
pw = 1000 # kg/m3 # density of water
ps = 1600 # kg/m3c # density of soil
g = 9.81  # m/s2 # force of gravity
yw = g*pw
ys = g*ps
phi = np.deg2rad(41)  # converts phi to radians ## set to 42 right now
z = df.Avg_Soil_Depth
#z = 2.5

# Slope stability variables
m = 1 # m # saturation ration (h/z)
l = 10 # m # length
w = 6.7 # m # width
C0 = 6400 # Pa
j = 1.5

#Define side/hollow slope range
# slope_ang = np.arange(27,51, 0.1) # Side slope range from 27 to 51 in degrees in 0.1 intervals
# slope_rad = np.deg2rad(slope_ang) # Side slope in radians
# hollow_rad = (np.arctan((0.8*(np.tan(np.deg2rad(slope_ang)))))) # Hollow slope in radians
# hollow_ang = np.rad2deg(np.arctan((0.8*(np.tan(np.deg2rad(slope_ang)))))) # Hollow slope in degrees
hollow_rad = np.deg2rad(df.Avg_Slope)


#Cohesion variables
Crb = C0*2.718281**(-z*j)
Crl = (C0/(j*z))*(1 - 2.718281**(-z*j))
# Crb = 5 # kPa # Cohesion of roots on base
# Crl = 5 # kPa # Lateral root cohesion

#%% Calculated variables for MDSTAB

# Earth Pressure Variables
K0 = 1-(np.sin(hollow_rad))

# Ka and Kp

aa = 8*(Crl/(ys*z))*(np.cos(hollow_rad)**2)*np.sin(phi)*np.cos(phi)

bb = 4*(Crl/(ys*z))*(np.cos(phi)**2)

cc = 4*(np.cos(hollow_rad)**2)*(((np.cos(hollow_rad)**2)-(np.cos(phi)**2)))

dd = 2*((Crl/(ys*z)))*(np.cos(phi)*np.sin(phi))

ee = 2*(np.cos(hollow_rad))**2

ff = 1/((np.cos(phi)**2))

Kp = ff * ((ee) + (dd) + ((cc) + (bb) + (aa))**0.5) - 1

Ka = ff * ((ee) + (dd) - ((cc) + (bb) + (aa))**0.5) - 1


#%% MDSTAB

# Define terms of equation
#Basal resistence force of the slide block
Frb = (Crb + ((np.cos(hollow_rad))**2)*z*(ys-yw*m)*np.tan(phi))*l*w

# Resisting force of each cross slope side of the slide block (2 lateral margins)
Frc = (Crl + (K0*0.5*z*(ys-yw*m**2)*np.tan(phi)))*(np.cos(hollow_rad)*z*l*2)

# Slope parallel component of passive force - slope-parallel component of active force
Frddu = (Kp-Ka)*0.5*(z**2)*(ys-yw*(m**2))*w

# Central block driving force
Fdc = (np.sin(hollow_rad))*(np.cos(hollow_rad))*z*ys*l*w


#Factor of Safety calculation
FS = (Frb + Frc + Frddu)/Fdc

df['FS'] = FS



plt.figure()
y_line_value = 1
# Loop through each buffer size and plot depth vs. time
for buffer_size in df['Buffer_Size'].unique():
    subset = df[df['Buffer_Size'] == buffer_size]  # Filter data for buffer size
    fs = subset.groupby("Year")["FS"].mean()  # Get avg depth per year

    plt.plot(fs.index, fs.values, marker='o', linestyle='-', label=f'Buffer {buffer_size}m')

# Labels and title
plt.xlabel("Year")
plt.ylabel("Factor of Safety")
plt.title("Soil Depth Over Time for Different Buffer Sizes")
plt.legend(title="Buffer Size (m)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.axhline(y=y_line_value, color='r', linestyle='--', linewidth=1.5, label=f"Threshold: {y_line_value}m")

