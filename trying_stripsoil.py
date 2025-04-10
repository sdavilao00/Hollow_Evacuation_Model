# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:32:06 2024

@author: Selina Davila Olivera
"""
#%%
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from osgeo import gdal
from landlab import imshowhs_grid
from landlab.components import TaylorNonLinearDiffuser
from landlab.io import read_esri_ascii, write_esri_ascii

#%%
# Input GeoTIFF file and directory
BASE_DIR = os.path.join(os.getcwd(), 'ExampleDEM')
INPUT_TIFF = 'mdstab_smooth_nowrap.tif'

# Setup output directory
OUT_DIR = os.path.join(BASE_DIR, 'simulation_results')
OUT_DIRpng = os.path.join(OUT_DIR, 'PNGs')
OUT_DIRtiff = os.path.join(OUT_DIR, 'GeoTIFFs')
OUT_DIRasc = os.path.join(OUT_DIR, 'ASCs') 
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIRpng, exist_ok=True)
os.makedirs(OUT_DIRtiff, exist_ok=True)
os.makedirs(OUT_DIRasc, exist_ok=True)

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
ft2mUS = 1200 / 3937   # US survey foot to meter conversion factor 
ft2mInt = 0.3048      # International foot to meter conversion factor 

import geopandas as gpd
from shapely.geometry import Point
from rasterio.features import geometry_mask

BUFFER_DISTANCE = 6  # Set the buffer distance in the same unit as the input DEM


def apply_buffer_to_soil_depth(grid, shapefile, buffer_distance, dem_path):
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask

    # Load the shapefile
    gdf = gpd.read_file(shapefile)

    # Open the DEM and extract CRS and transform
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        transform = src.transform
        out_shape = src.read(1).shape

    # Reproject shapefile to DEM CRS if needed
    if gdf.crs != dem_crs:
        print(f"Reprojecting shapefile from {gdf.crs} to match DEM CRS...")
        gdf = gdf.to_crs(dem_crs)
    else:
        print("Shapefile already in the same CRS as DEM.")

    # Create buffer around each point
    buffered_geoms = gdf.buffer(buffer_distance)

    # Create mask from buffered geometry
    mask = geometry_mask(
        [buffered_geoms.unary_union],
        transform=transform,
        invert=True,
        out_shape=out_shape
    )

    # Apply mask to soil__depth
    soil_depth = grid.at_node['soil__depth']
    soil_depth[mask.flatten()] = 0.0

    print("Applied 0m soil depth to buffer zones.")
    return grid
    
    
    # Plot the Mask
    plt.figure(figsize=(6, 5))
    plt.imshow(mask, cmap="gray", origin="upper")
    plt.colorbar(label="Buffer Mask (0 = Unmasked, 1 = Masked)")
    plt.title("Buffer Mask Visualization")
    plt.show()
    
    # Plot Soil Depth Grid with Buffer Overlay
    fig, ax = plt.subplots(figsize=(6, 5))
    soil_depth_reshaped = soil_depth.reshape(grid.shape)
    im = ax.imshow(soil_depth_reshaped, cmap="viridis", origin="upper")
    plt.colorbar(im, label="Soil Depth (m)")
    gdf.plot(ax=ax, marker="o", color="red", markersize=5, label="Buffer Points")
    plt.legend()
    plt.title("Soil Depth with Buffer Overlay")
    plt.xlabel("X Grid")
    plt.ylabel("Y Grid")
    plt.show()
    
    return grid
#%%
# asc_file = os.path.join(BASE_DIR, 'mdstab_smooth_nowrap.asc')  
# shapefile_path = os.path.join(BASE_DIR, 'MDSTAB_hollow.shp')  
# dem_path = os.path.join(BASE_DIR, 'mdstab_smooth_nowrap.tif')  # Use the original DEM
# buffer_distance = 6

# grid, _ = read_esri_ascii(asc_file, name='topographic__elevation')

# # Apply buffer with the DEM file path
# grid = apply_buffer_to_soil_depth(grid, shapefile_path, buffer_distance, dem_path)

# # Plot to verify
# plt.figure(figsize=(6, 5))
# imshowhs_grid(grid, grid.at_node['soil__depth'], plot_type="Hillshade")
# plt.colorbar(label="Soil Depth (m)")
# plt.title("Soil Depth After Buffer Application")
# plt.show()

#%%

# Modify init_simulation to include buffer functionality
def init_simulation(asc_file, K, Sc, XYZunit=None, shapefile=None, buffer_distance=BUFFER_DISTANCE, dem_path=None):
    grid, _ = read_esri_ascii(asc_file, name='topographic__elevation')
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)
    
    # Initialize soil depth everywhere to 0.5m
    soil_depth = np.full(grid.number_of_nodes, 0.5)
    grid.add_field('soil__depth', soil_depth, at='node')
    
    # Apply 0m buffer zones
    if shapefile and dem_path:
        grid = apply_buffer_to_soil_depth(grid, shapefile, buffer_distance, dem_path)

    # Diffusivity setup
    if XYZunit is None or 'meter' in XYZunit.lower():
        Kc = K
    elif 'foot' in XYZunit.lower():
        Kc = K / (ft2mUS ** 2) if "US" in XYZunit else K / (ft2mInt ** 2)
    else:
        raise RuntimeError("Unsupported unit type for K conversion.")

    TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=Kc, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable="pass")
    return grid, TNLD


#%%
def plot_save(grid, z, basefilename, time, K, mean_res=None, XYZunit=None):
    plt.figure(figsize=(6, 5.25))
    imshowhs_grid(grid, z, plot_type="Hillshade")
    plt.title(f"{basefilename} \n Time: {time} years (K = {K} m$^{{2}}$/yr)", fontsize='small', fontweight="bold")
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')
    if XYZunit is not None:
        plt.xlabel(f'X-axis grids \n(grid size ≈ {round(mean_res, 4)} [{XYZunit}])', fontsize='small')
        plt.ylabel(f'Y-axis grids \n(grid size ≈ {round(mean_res, 4)} [{XYZunit}])', fontsize='small')
    else:
        print("The function assumes the input XYZ units are in meters.")
        plt.xlabel(f'X-axis grids \n(grid size ≈ {1.0} [meters])', fontsize='small')
        plt.ylabel(f'Y-axis grids \n(grid size ≈ {1.0} [meters])', fontsize='small') 
    plt.grid(False)
    plt.tight_layout()

    # Save PNG file
    png_path = os.path.join(OUT_DIRpng, f"{basefilename}_{time}yrs_(K={K}).png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"'{png_path}' saved to 'simulation_results' folder")

    # Write the grid to an ASC file
    asc_path = os.path.join(OUT_DIRasc, f"{basefilename}_{time}_(K={K})yrs.asc")
    write_esri_ascii(asc_path, grid, names=['topographic__elevation'], clobber=True)
    print(f"'{asc_path}' created and saved.")
    
    return asc_path

#%%
def plot_change(change, title, basefilename, time, K, grid_shape, flip_vertical= True, flip_horizontal=False):
    """Plot and save PNGs of changes with optional flipping."""
    # Check if change is 1D and reshape it if necessary
    if change.ndim == 1:
        change = change.reshape(grid_shape)  # Ensure this matches your grid's dimensions

    # Flip the data vertically if specified
    if flip_vertical:
        change = np.flipud(change)

    # Flip the data horizontally if specified
    if flip_horizontal:
        change = np.fliplr(change)
        

    plt.figure(figsize=(6, 5.25))
    plt.imshow(change, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Change Value')
    plt.title(f"{title} \n Time: {time} years (K = {K} m$^{{2}}$/yr)", fontsize='small', fontweight="bold")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(False)
    plt.tight_layout()

    png_change_path = os.path.join(OUT_DIRpng, f"{basefilename}_{title.lower().replace(' ', '_')}_{time}yrs_(K={K}).png")
    plt.savefig(png_change_path, dpi=150)
    plt.close()
    print(f"'{png_change_path}' saved to 'simulation_results' folder")

#%%
def save_as_tiff(data, filename, meta, grid_shape, flip_vertical=True, flip_horizontal=False):
    """Save a numpy array as a GeoTIFF file with optional flipping."""
    # Check if data is 1D and reshape it if necessary
    if data.ndim == 1:
        data = data.reshape(grid_shape)  # Ensure this matches your grid's dimensions

    # Flip the data vertically if specified
    if flip_vertical:
        data = np.flipud(data)

    # Flip the data horizontally if specified
    if flip_horizontal:
        data = np.fliplr(data)
        
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    print(f"'{os.path.basename(filename)}' saved as a GeoTIFF.")


#%%
def run_simulation(in_tiff, K, Sc, dt, target_time, shapefile):
    basefilename = os.path.splitext(in_tiff)[0]
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(os.path.join(BASE_DIR, in_tiff), in_asc)

    grid, tnld = init_simulation(in_asc, K, Sc, XYZunit)
    z_old = grid.at_node['topographic__elevation']
    soil_depth = grid.at_node['soil__depth']
    
    # Store the initial soil depth
    initial_soil_depth = soil_depth.copy()
    
    # Initialize total soil depth
    total_soil_depth = soil_depth.copy()


    with rasterio.open(os.path.join(BASE_DIR, in_tiff)) as src:
        meta = src.meta.copy()

    asc_path = plot_save(grid, z_old, basefilename, 0, K, mean_res, XYZunit)


    num_steps = int(target_time / dt)
    for i in range(num_steps):
        tnld.run_one_step(dt)
        time = (i + 1) * dt
        
        
        # Calculate soil production for this time step
        production_rate = (pr / ps) * (P0 * np.exp(-total_soil_depth/h0)) * dt
        z_new = grid.at_node['topographic__elevation']
        elevation_change = z_new - z_old
        
        # # Calculate soil production and add to total
        # production_rate = (pr / ps) * (P0 * np.exp(-total_soil_depth / h0)) * dt
        # total_soil_produced_nodes = production_rate
        
        # Calculate soil transport (erosion) and add to total
        # soil_transport_nodes = np.abs(elevation_change[elevation_change < 0])  # Sum of erosion (negative elevation change

        # Check if erosion exceeds the current total soil depth
        erosion_exceeds = abs(elevation_change) > initial_soil_depth

        if np.any(erosion_exceeds):
            # For nodes where erosion exceeds soil depth
            # Set the new elevation to initial elevation minus current total soil depth
            z_new[erosion_exceeds] = z_old[erosion_exceeds] - total_soil_depth[erosion_exceeds]
            
            # Set soil depth to zero where erosion exceeds initial soil depth
            total_soil_depth[erosion_exceeds] = production_rate[erosion_exceeds]
            
            print(f"Erosion exceeded soil depth at time {time} years. Adjusting elevation.")

        # Initialize change in soil depth with production rate
        change_in_soil_depth = production_rate.copy()

        # For nodes where erosion does NOT exceed initial soil depth, update as normal
        change_in_soil_depth[~erosion_exceeds] = elevation_change[~erosion_exceeds] + production_rate[~erosion_exceeds]

        # Update total soil depth
        total_soil_depth = np.where(erosion_exceeds, production_rate, total_soil_depth + change_in_soil_depth)

        # Ensure no negative soil depth as a final check
        # total_soil_depth = np.maximum(total_soil_depth, 0)

        # Update the soil depth field in the grid
        grid.at_node['soil__depth'] = total_soil_depth

        # Copy the new elevation for the next iteration
        z_old = z_new.copy()

        # Output results every 1000 years
        if time % 50 == 0:
            print(f"Results at {time} years:")
            print(f"Total Soil Depth: {total_soil_depth}")
            print(f"Elevation: {z_new}")
            print(f"Change in Elevation: {elevation_change}")
            print(f"Change in Soil Depth: {change_in_soil_depth}")

            # Save change in elevation and soil depth as TIFFs
            tiff_elevation_change_path = os.path.join(OUT_DIRtiff, f"{basefilename}_change_in_elevation_{time}yrs.tif")
            save_as_tiff(elevation_change, tiff_elevation_change_path, meta, grid.shape)

            tiff_soil_depth_change_path = os.path.join(OUT_DIRtiff, f"{basefilename}_change_in_soil_depth_{time}yrs.tif")
            save_as_tiff(change_in_soil_depth, tiff_soil_depth_change_path, meta, grid.shape)

            tiff_total_soil_depth_path = os.path.join(OUT_DIRtiff, f"{basefilename}_total_soil_depth_{time}yrs.tif")
            save_as_tiff(total_soil_depth, tiff_total_soil_depth_path, meta, grid.shape)
            
            tiff_production_rate_path = os.path.join(OUT_DIRtiff, f"{basefilename}_production_rate_{time}yrs.tif")
            save_as_tiff(production_rate, tiff_production_rate_path, meta, grid.shape)

            # Save the current elevation as a GeoTIFF
            tiff_elevation_path = os.path.join(OUT_DIRtiff, f"{basefilename}_elevation_{time}yrs.tif")
            asc_to_tiff(asc_path, tiff_elevation_path, meta)
            
            # Save change in elevation and soil depth as PNGs
            grid_shape = grid.shape  # Get the shape of the grid
            plot_change(elevation_change, "Change in Elevation", basefilename, time, K, grid_shape)
            plot_change(change_in_soil_depth, "Change in Soil Depth", basefilename, time, K, grid_shape)
            plot_change(total_soil_depth, "Total Soil Depth", basefilename, time, K, grid_shape)
            plot_change(production_rate, "Soil Produced", basefilename, time, K, grid_shape)
         

        if time % 50 == 0:
            asc_path = plot_save(grid, z_new, basefilename, time, K, mean_res, XYZunit)
            tiff_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs_(K={K}).tif")
            asc_to_tiff(asc_path, tiff_path, meta)


    in_prj = in_asc.replace('.asc', '.prj')
    os.remove(in_asc)
    os.remove(in_prj)
    print("Simulation completed. Temporary ASCII & PRJ files cleaned up.")


#%%
# Example run parameters
K = 0.0042  # Diffusion coefficient
Sc = 1.25  # Critical slope gradient
pr = 2000   # Ratio of production (example value)
ps = 1000   # Ratio of soil loss (example value)
P0 = 0.0003  # Initial soil production rate (example value, e.g., kg/m²/year)
h0 = 0.4   # Depth constant related to soil production (example value, e.g., meters)
dt = 50
target_time = 300

shapefile_path = os.path.join(BASE_DIR, 'MDSTAB_hollow_buff.shp')
run_simulation(INPUT_TIFF, K, Sc, dt, target_time, shapefile=shapefile_path)

#%%

import os
import glob

# Define input folder and output folder
input_folder = "C:/Users/sdavilao/OneDrive - University Of Oregon/Documents/3D_test/ExampleDEM/simulation_results/GeoTIFFS"
output_folder = "C:/Users/sdavilao/OneDrive - University Of Oregon/Documents/3D_test/ExampleDEM/simulation_results/GeoTIFFS/reprojected/"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define target CRS
target_crs = "EPSG:32610"

# Get all .tif files in the input folder
tif_files = glob.glob(os.path.join(input_folder, "*.tif"))

# Loop through each file and reproject
for input_tif in tif_files:
    # Define output file path
    filename = os.path.basename(input_tif)  # Get the file name
    output_tif = os.path.join(output_folder, filename)  # Save in output folder

    # Perform the reprojection
    gdal.Warp(output_tif, input_tif, dstSRS=target_crs, resampleAlg=gdal.GRA_NearestNeighbour)

    print(f"Reprojected: {filename} -> {output_tif}")

print("Batch reprojection complete!")




