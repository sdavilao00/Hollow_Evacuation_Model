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
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

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
buffer_sizes = [2, 4, 6, 7, 8, 10, 12, 14, 16]

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

#%% ---------------- INTERPOLATE FS TO FIND EXACT FS = 1 YEAR ----------------
optimal_buffers = {}

for point_id in df['Point_ID'].unique():
    point_data = df[df['Point_ID'] == point_id]  # Get data for this Point_ID

    # Step 1: Loop through each buffer size and interpolate FS over time
    buffer_first_crossings = {}

    for buffer_size in point_data['Buffer_Size'].unique():
        buffer_data = point_data[point_data['Buffer_Size'] == buffer_size].sort_values(by='Year')

        # Get Year and FS values as arrays
        years = buffer_data['Year'].values
        fs_values = buffer_data['FS'].values

        # Only interpolate if there are at least two points
        if len(years) < 2:
            continue

        # Interpolate FS over time
        fs_interp = interp1d(fs_values, years, kind='linear', bounds_error=False, fill_value='extrapolate')

        # Estimate the exact year where FS = 1
        estimated_year = fs_interp(1)  # Find year where FS crosses 1

        if not np.isnan(estimated_year) and estimated_year > min(years) and estimated_year < max(years):
            buffer_first_crossings[buffer_size] = estimated_year

    # Step 2: Find the buffer that reaches FS = 1 the fastest (smallest interpolated year)
    if not buffer_first_crossings:
        print(f"Warning: No FS = 1 data for Point_ID {point_id}. Skipping.")
        continue

    optimal_buffer_size = min(buffer_first_crossings, key=buffer_first_crossings.get)
    optimal_year = buffer_first_crossings[optimal_buffer_size]

    # Step 3: Interpolate Soil Depth and Slope at this interpolated year
    selected_buffer_data = point_data[point_data["Buffer_Size"] == optimal_buffer_size].sort_values(by='Year')

    # Interpolating Soil Depth and Slope using the same approach
    soil_depth_interp = interp1d(selected_buffer_data['Year'], selected_buffer_data['Avg_Soil_Depth'], kind='linear', bounds_error=False, fill_value='extrapolate')
    slope_interp = interp1d(selected_buffer_data['Year'], selected_buffer_data['Avg_Slope'], kind='linear', bounds_error=False, fill_value='extrapolate')

    estimated_soil_depth = soil_depth_interp(optimal_year)
    estimated_slope = slope_interp(optimal_year)

    # Step 4: Store interpolated values
    optimal_buffers[point_id] = {
        'Optimal_Buffer': optimal_buffer_size,  # Buffer that reached FS = 1 first
        'Year': optimal_year,  # Interpolated Year where FS = 1
        'FS': 1.0,  # Exact FS value
        'Avg_Soil_Depth': estimated_soil_depth,  # Interpolated soil depth
        'Avg_Slope': estimated_slope  # Interpolated slope
    }

# Convert to DataFrame
df_optimal_interpolated = pd.DataFrame.from_dict(optimal_buffers, orient='index')
df_optimal_interpolated.reset_index(inplace=True)
df_optimal_interpolated.rename(columns={'index': 'Point_ID'}, inplace=True)

# Save to CSV file
output_path = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Desktop\QGIS\HC\04282023\optimal_buffer_results_interpolated.csv"
df_optimal_interpolated.to_csv(output_path, index=False)

# Print Results
print(df_optimal_interpolated)





#%%

# Define the target year range for interpolation
min_year = df["Year"].min()
max_year = df["Year"].max()
target_years = np.arange(min_year, max_year + 1, 1)  # Interpolating yearly

# List to store interpolated results
interpolated_results = []

#%% ---------------- INTERPOLATE FS, SOIL DEPTH, & SLOPE ACROSS ALL YEARS ----------------
for point_id in df['Point_ID'].unique():
    point_data = df[df['Point_ID'] == point_id]

    for buffer_size in point_data['Buffer_Size'].unique():
        buffer_data = point_data[point_data["Buffer_Size"] == buffer_size].sort_values(by='Year')

        # Ensure at least two points exist for interpolation
        if len(buffer_data) < 2:
            continue

        # Interpolate FS, Soil Depth, and Slope
        fs_interp = interp1d(buffer_data["Year"], buffer_data["FS"], kind='linear', bounds_error=False, fill_value='extrapolate')
        soil_depth_interp = interp1d(buffer_data["Year"], buffer_data["Avg_Soil_Depth"], kind='linear', bounds_error=False, fill_value='extrapolate')
        slope_interp = interp1d(buffer_data["Year"], buffer_data["Avg_Slope"], kind='linear', bounds_error=False, fill_value='extrapolate')

        # Compute interpolated values for all target years
        interpolated_fs = fs_interp(target_years)
        interpolated_soil_depth = soil_depth_interp(target_years)
        interpolated_slope = slope_interp(target_years)

        # Store results
        for year, fs, depth, slope in zip(target_years, interpolated_fs, interpolated_soil_depth, interpolated_slope):
            interpolated_results.append({
                'Point_ID': point_id,
                'Buffer_Size': buffer_size,
                'Year': year,
                'FS': fs,
                'Avg_Soil_Depth': depth,
                'Avg_Slope': slope
            })

# Convert to DataFrame
df_interpolated = pd.DataFrame(interpolated_results)

# Save to CSV file
output_path = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Desktop\QGIS\HC\04282023\interpolated_dataset.csv"
df_interpolated.to_csv(output_path, index=False)

# Print preview of the interpolated DataFrame
print(df_interpolated.head())


#%% ---------------- EXTRACT FULL FS THROUGH TIME DATASET ----------------

# Select only necessary columns from df_interpolated
df_fs_through_time = df_interpolated[['Point_ID', 'Buffer_Size', 'Year', 'FS']].copy()

# Save to CSV
output_path = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Desktop\QGIS\HC\04282023\fs_through_time.csv"
df_fs_through_time.to_csv(output_path, index=False)

# Print preview
print(df_fs_through_time.head())




#%% ---------------- EXTRACT FS AT FAILURE YEAR (FROM df_optimal_interpolated) ----------------
optimal_fs_values = []  # List to store results

for point_id in df_interpolated['Point_ID'].unique():
    # Get the failure year from df_optimal_interpolated
    optimal_year_row = df_optimal_interpolated[df_optimal_interpolated["Point_ID"] == point_id]

    if optimal_year_row.empty:
        continue  # Skip if no optimal year is found

    optimal_year = optimal_year_row["Year"].values[0]  # Extract the correct failure year

    # Get FS values for all buffers at this optimal year
    point_data = df_interpolated[df_interpolated["Point_ID"] == point_id]

    for buffer_size in point_data["Buffer_Size"].unique():
        buffer_data = point_data[point_data["Buffer_Size"] == buffer_size].sort_values(by="Year")

        if buffer_data.empty:
            continue  # Skip if no data for this buffer

        # Check if exact year exists in the dataset
        exact_year_data = buffer_data[buffer_data["Year"] == optimal_year]

        if not exact_year_data.empty:
            # Use exact values if they exist
            best_year_row = exact_year_data.iloc[0]
            fs_value = best_year_row["FS"]
        else:
            # Use linear interpolation if the exact year is missing
            if len(buffer_data) < 2:
                continue  # Skip if there's not enough data for interpolation

            fs_value = np.interp(optimal_year, buffer_data["Year"], buffer_data["FS"])

        # Store results
        optimal_fs_values.append({
            'Point_ID': point_id,
            'Buffer_Size': buffer_size,
            'Year': optimal_year,  # The failure year pulled from df_optimal_interpolated
            'FS': fs_value
        })

# Convert to DataFrame
df_fs_at_optimal_year = pd.DataFrame(optimal_fs_values)

# Save to CSV
output_path = r"C:\Users\sdavilao\OneDrive - University Of Oregon\Desktop\QGIS\HC\04282023\fs_at_optimal_year.csv"
df_fs_at_optimal_year.to_csv(output_path, index=False)

# Print Results
print(df_fs_at_optimal_year.head())




# Choose a specific Point_ID for plotting
selected_point_id = 2  # Change this to analyze another Point_ID

# Filter dataset for the selected Point_ID
df_filtered = df_fs_at_optimal_year[df_fs_at_optimal_year["Point_ID"] == selected_point_id]

plt.figure(figsize=(8, 6))

sns.scatterplot(data=df_filtered, x='Buffer_Size', y='FS', marker='o', color='b', s=100, edgecolor='black')

# Labels and Title
plt.xlabel("Buffer Size (m)")
plt.ylabel("Factor of Safety (FS)")
plt.title(f"FS Across Buffer Sizes at Failure Year (From df_optimal_interpolated) for Point_ID {selected_point_id}")
plt.axhline(y=1, color='r', linestyle='--', linewidth=1.5, label="FS = 1 Threshold")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()


