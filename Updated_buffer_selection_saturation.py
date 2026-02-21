# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:24:25 2025

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
from rasterstats import zonal_stats

#%% ---------------- LOAD & PROCESS DATA ----------------
# Define file paths
shapefile_path = r"C:\Users\sdavilao\Documents\newcodesoil\reproj_shp\ext22_32610.shp"
dem_path = r"C:\Users\sdavilao\Documents\newcodesoil\dem_smooth_m_warp.tif"
slope_path = r"C:\Users\sdavilao\Documents\newcodesoil\slope_smooth_m_warp.tif"
downslope_lines =r"C:\Users\sdavilao\Documents\newcodesoil\polylines\ext22_lines.shp" 

# Define soil depth raster directory and naming pattern
soil_depth_dir = r"C:\Users\sdavilao\Documents\newcodesoil\simulation_results\new\GeoTIFFs\reproj_tif"
basename = "ext22"
soil_depth_pattern = os.path.join(soil_depth_dir, f"{basename}_total_soil_depth_*yrs_32610.tif")

# Load the point shapefile
gdf = gpd.read_file(shapefile_path)

line_gdf = gpd.read_file(downslope_lines)
line_gdf = line_gdf.to_crs(gdf.crs)  # Ensure same CRS

if 'id' in line_gdf.columns:
    line_gdf = line_gdf.rename(columns={'id': 'Point_ID'})


# Rename 'id' column to 'Point_ID' if it exists
if 'id' in gdf.columns:
    gdf = gdf.rename(columns={'id': 'Point_ID'})

# Ensure the shapefile and raster CRS match
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs  
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)  

# Get all soil depth .tif files and sort them by year
soil_depth_files = sorted(glob.glob(soil_depth_pattern))

# Define buffer sizes to test
buffer_sizes = [4,5,6,7,8,9,10] ##this should be bounded??

# Create a list to store results
results = []

point_slope_dict = {}
point_slope_count = {}

# Loop through each point in the shapefile
for _, line_row in line_gdf.iterrows():
    pid = line_row["Point_ID"]
    try:
        zs = zonal_stats([line_row["geometry"]], slope_path, stats=["count", "mean"], nodata=-9999)
        avg = zs[0]["mean"]
        count = zs[0]["count"]
        point_slope_dict[pid] = avg
        point_slope_count[pid] = count
        print(f"[OK] Slope for Point {pid}: {avg:.4f} from {count} pixels")
    except Exception as e:
        point_slope_dict[pid] = np.nan
        point_slope_count[pid] = 0
        print(f"[FAIL LINE SLOPE] Point {pid}: {e}")

# Now begin main loop for each point and buffer
for _, row in gdf.iterrows():
    point_geom = row.geometry
    point_id = row['Point_ID']

    for buffer_distance in buffer_sizes:
        buffer_geom = point_geom.buffer(buffer_distance)
        buffer_json = [buffer_geom.__geo_interface__]
        print(f"\n[CHECK] Point {point_id}, Buffer {buffer_distance}m")

        # Use precomputed slope
        avg_slope = point_slope_dict.get(point_id, np.nan)
        slope_count = point_slope_count.get(point_id, 0)
        print(f"    → Constant slope used: {avg_slope}, pixel count: {slope_count}")

        ### Step 2: Extract Soil Depth for Each Year ###
        for soil_depth_tif in soil_depth_files:
            match = re.search(r'_(\d+)yrs.*\.tif$', soil_depth_tif)
            if match:
                year = int(match.group(1))
            else:
                continue

            try:
                print(f"🔎 Opening soil raster: {os.path.basename(soil_depth_tif)}")
                with rasterio.open(soil_depth_tif) as soil_depth_raster:
                    soil_depth_image, _ = mask(soil_depth_raster, buffer_json, crop=True)
                    soil_depth_image = soil_depth_image[0]
                    nodata_val = soil_depth_raster.nodata
                    print(f"    → Raster bounds: {soil_depth_raster.bounds}")
                    print(f"    → Buffer bounds: {buffer_geom.bounds}")
                    print(f"    → Nodata value: {nodata_val}")
                    print(f"    → Min/Max in masked soil: {soil_depth_image.min()}, {soil_depth_image.max()}")

                    valid_soil_depth_values = soil_depth_image[
                        (soil_depth_image != nodata_val) & (soil_depth_image > 0)]
                    print(f"    → Valid values count: {valid_soil_depth_values.size}")

                    if valid_soil_depth_values.size == 0:
                        print(f"⚠️ No valid soil depth at Year {year}, Point {point_id}, Buffer {buffer_distance}")
                        avg_soil_depth = np.nan
                    else:
                        avg_soil_depth = np.mean(valid_soil_depth_values)
                        print(f"    → Avg soil depth: {avg_soil_depth:.4f}")

            except Exception as e:
                print(f"[FAIL SOIL] Year {year}, Point {point_id}, Buffer {buffer_distance}: {e}")
                avg_soil_depth = np.nan
                continue  # skip bad year

            # Store result regardless of validity for debugging
            results.append({
                'Point_ID': point_id,
                'Year': year,
                'Buffer_Size': buffer_distance,
                'Avg_Slope': avg_slope,
                'Avg_Soil_Depth': avg_soil_depth
            })
            print(f"✅ Stored: Point {point_id}, Year {year}, Buffer {buffer_distance}")

# Final reporting and sorting
print(f"\n🔍 Total results collected: {len(results)}")
df = pd.DataFrame(results)

if not df.empty and 'Point_ID' in df.columns:
    df = df.sort_values(by=['Point_ID', 'Year', 'Buffer_Size']).reset_index(drop=True)
else:
    print("⚠️ No data to sort — check for skipped buffers or missing raster overlap.")
#%% ---------------- CALCULATE FACTOR OF SAFETY ----------------
# Constants 
Sc = 1.25  
pw = 1000  
ps = 1600  
g = 9.81  
yw = g * pw
ys = g * ps
phi = np.deg2rad(41)  
m = 0.85
l = 10  
w = 6.7  
C0 = 760  
j = 0.8 

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

    # Kp = ff * ((ee) + (dd) + ((cc) + (bb) + (aa)) ** 0.5) - 1
    # Ka = ff * ((ee) + (dd) - ((cc) + (bb) + (aa)) ** 0.5) - 1
    
    Kp = np.tan((np.deg2rad(45))+(phi/2))**2
    Ka = np.tan((np.deg2rad(45))-(phi/2))**2


    Frb = (Crb + ((np.cos(hollow_rad)) ** 2) * z * (ys - yw * m) * np.tan(phi)) * l * w
    Frc = (Crl + (K0 * 0.5 * z * (ys - yw * m ** 2) * np.tan(phi))) * (np.cos(hollow_rad) * z * l * 2)
    Frddu = (Kp - Ka) * 0.5 * (z ** 2) * (ys - yw * (m ** 2)) * w
    Fdc = (np.sin(hollow_rad)) * (np.cos(hollow_rad)) * z * ys * l * w

    return (Frb + Frc + Frddu) / Fdc if Fdc != 0 else np.nan

df['FS'] = df.apply(calculate_fs, axis=1)

# ---------------- INTERPOLATE FS TO FIND EXACT FS = 1 YEAR ----------------
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
        estimated_year = fs_interp(1.00000)  # Find year where FS crosses 1

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
        'Optimal_Buffer_m': optimal_buffer_size,  # Buffer that reached FS = 1 first
        'Year': optimal_year,  # Interpolated Year where FS = 1
        'FS': 1.0,  # Exact FS value
        'Avg_Soil_Depth_m': estimated_soil_depth,  # Interpolated soil depth
        'Avg_Slope_deg': estimated_slope  # Interpolated slope
    }

# Convert to DataFrame
df_optimal_interpolated = pd.DataFrame.from_dict(optimal_buffers, orient='index')
df_optimal_interpolated.reset_index(inplace=True)
df_optimal_interpolated.rename(columns={'index': 'Point_ID'}, inplace=True)

# # Save to CSV file
output_path = r"C:\Users\sdavilao\Documents\newcodesoil\results\new\m\optimal_buffer_results_interpolated_ext1_760_mp5.csv"
df_optimal_interpolated.to_csv(output_path, index=False)

# Print Results
print(df_optimal_interpolated)
#%%
# Constants 
Sc = 1.25  
pw = 1000  
ps = 1600  
g = 9.81  
yw = g * pw
ys = g * ps
phi = np.deg2rad(41)  
m = 0.5  
l = 10  
w = 6.7  
C0 = 1920  
j = 0.8 

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

    # Kp = ff * ((ee) + (dd) + ((cc) + (bb) + (aa)) ** 0.5) - 1
    # Ka = ff * ((ee) + (dd) - ((cc) + (bb) + (aa)) ** 0.5) - 1
    
    Kp = np.tan((np.deg2rad(45))+(phi/2))**2
    Ka = np.tan((np.deg2rad(45))-(phi/2))**2


    Frb = (Crb + ((np.cos(hollow_rad)) ** 2) * z * (ys - yw * m) * np.tan(phi)) * l * w
    Frc = (Crl + (K0 * 0.5 * z * (ys - yw * m ** 2) * np.tan(phi))) * (np.cos(hollow_rad) * z * l * 2)
    Frddu = (Kp - Ka) * 0.5 * (z ** 2) * (ys - yw * (m ** 2)) * w
    Fdc = (np.sin(hollow_rad)) * (np.cos(hollow_rad)) * z * ys * l * w

    return (Frb + Frc + Frddu) / Fdc if Fdc != 0 else np.nan

df['FS'] = df.apply(calculate_fs, axis=1)

# ---------------- INTERPOLATE FS TO FIND EXACT FS = 1 YEAR ----------------
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
        'Optimal_Buffer_m': optimal_buffer_size,  # Buffer that reached FS = 1 first
        'Year': optimal_year,  # Interpolated Year where FS = 1
        'FS': 1.0,  # Exact FS value
        'Avg_Soil_Depth_m': estimated_soil_depth,  # Interpolated soil depth
        'Avg_Slope_deg': estimated_slope  # Interpolated slope
    }

# Convert to DataFrame
df_optimal_interpolated = pd.DataFrame.from_dict(optimal_buffers, orient='index')
df_optimal_interpolated.reset_index(inplace=True)
df_optimal_interpolated.rename(columns={'index': 'Point_ID'}, inplace=True)

# # Save to CSV file
output_path = r"C:\Users\sdavilao\Documents\newcodesoil\results\new\m\optimal_buffer_results_interpolated_ext1_1920_mp5.csv"
df_optimal_interpolated.to_csv(output_path, index=False)

# Print Results
print(df_optimal_interpolated)
#%%
# Constants 
Sc = 1.25  
pw = 1000  
ps = 1600  
g = 9.81  
yw = g * pw
ys = g * ps
phi = np.deg2rad(41)  
m = 0.5
l = 10  
w = 6.7  
C0 = 3600  
j = 0.8 

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

    # Kp = ff * ((ee) + (dd) + ((cc) + (bb) + (aa)) ** 0.5) - 1
    # Ka = ff * ((ee) + (dd) - ((cc) + (bb) + (aa)) ** 0.5) - 1
    
    Kp = np.tan((np.deg2rad(45))+(41/2))**2
    Ka = np.tan((np.deg2rad(45))-(41/2))**2


    Frb = (Crb + ((np.cos(hollow_rad)) ** 2) * z * (ys - yw * m) * np.tan(phi)) * l * w
    Frc = (Crl + (K0 * 0.5 * z * (ys - yw * m ** 2) * np.tan(phi))) * (np.cos(hollow_rad) * z * l * 2)
    Frddu = (Kp - Ka) * 0.5 * (z ** 2) * (ys - yw * (m ** 2)) * w
    Fdc = (np.sin(hollow_rad)) * (np.cos(hollow_rad)) * z * ys * l * w

    return (Frb + Frc + Frddu) / Fdc if Fdc != 0 else np.nan

df['FS'] = df.apply(calculate_fs, axis=1)

# ---------------- INTERPOLATE FS TO FIND EXACT FS = 1 YEAR ----------------
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
        'Optimal_Buffer_m': optimal_buffer_size,  # Buffer that reached FS = 1 first
        'Year': optimal_year,  # Interpolated Year where FS = 1
        'FS': 1.0,  # Exact FS value
        'Avg_Soil_Depth_m': estimated_soil_depth,  # Interpolated soil depth
        'Avg_Slope_deg': estimated_slope  # Interpolated slope
    }

# Convert to DataFrame
df_optimal_interpolated = pd.DataFrame.from_dict(optimal_buffers, orient='index')
df_optimal_interpolated.reset_index(inplace=True)
df_optimal_interpolated.rename(columns={'index': 'Point_ID'}, inplace=True)

# # Save to CSV file
output_path = r"C:\Users\sdavilao\Documents\newcodesoil\results\new\m\optimal_buffer_results_interpolated_ext188_3600_mp5.csv"
df_optimal_interpolated.to_csv(output_path, index=False)

# Print Results
print(df_optimal_interpolated)

