# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:32:06 2024

@author: 12092
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
INPUT_TIFF = 'hc_smooth.tif'

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
        XYZunit = src.crs.linear_units   # assuming the unit of XYZ direction are the same
        mean_res = np.mean(src.res)

    gdal.Translate(out_path, in_path, format='AAIGrid', xRes=mean_res, yRes=mean_res)
    print(f"The input GeoTIFF is temporarily converted to '{os.path.basename(out_path)}' with grid spacing {mean_res} ({XYZunit})")
    return mean_res, XYZunit

def asc_to_tiff(asc_path, tiff_path, meta):
    data = np.loadtxt(asc_path, skiprows=10)
    meta.update(dtype=rasterio.float32, count=1, compress='deflate')

    with rasterio.open(tiff_path, 'w', **meta) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    print(f"'{os.path.basename(tiff_path)}' saved to 'simulation_results' folder")   

#%%
ft2mUS = 1200 / 3937   # US survey foot to meter conversion factor 
ft2mInt = 0.3048      # International foot to meter conversion factor 

def init_simulation(asc_file, K, Sc, XYZunit=None):
    grid, _ = read_esri_ascii(asc_file, name='topographic__elevation')
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)  # boundaries open: allowing sediment move in/out of the study area

    # Create a soil depth grid with a uniform depth of 0.5m
    soil_depth = np.full(grid.number_of_nodes, 0.5)  # Soil depth of 0.5m at each node
    grid.add_field('soil__depth', soil_depth, at='node')

    # Check the unit of XYZ and make unit conversion of K when needed
    if XYZunit is None:
        print("The function assumes the input XYZ units are in meters.")
        TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=K, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable="pass")
    elif XYZunit is not None:
        if any(unit in XYZunit.lower() for unit in ["metre".lower(), "meter".lower()]):
            print("Input XYZ units are in meters. No unit conversion is made")
            TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=K, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable="pass")
        elif any(unit in XYZunit.lower() for unit in ["foot".lower(), "feet".lower(), "ft".lower()]):  
            if any(unit in XYZunit.lower() for unit in ["US".lower(), "United States".lower()]):
                print("Input XYZ units are in US survey feet. A unit conversion to meters is made for K")
                Kc = K / (ft2mUS ** 2)
                TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=Kc, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable="pass")
            else:
                print("Input XYZ units are in international feet. A unit conversion to meters is made for K")
                Kc = K / (ft2mInt ** 2)
                TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=Kc, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable="pass")
        else:
            message = (
            "WARNING: The code execution is stopped. "
            "The input XYZ units must be in feet or meters."
            )
            raise RuntimeError(message)

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
def plot_change(change, title, basefilename, time, K, grid_shape):
    # Check if change is 1D and reshape it if necessary
    if change.ndim == 1:
        change = change.reshape(grid_shape)  # Ensure this matches your grid's dimensions

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
def save_as_tiff(data, filename, meta, grid_shape):
    """Save a numpy array as a GeoTIFF file."""
    # Check if data is 1D and reshape it if necessary
    if data.ndim == 1:
        data = data.reshape(grid_shape)  # Ensure this matches your grid's dimensions

    # Flip the data vertically
    data = np.flipud(data)

    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    print(f"'{os.path.basename(filename)}' saved as a GeoTIFF.")


#%%
def run_simulation(in_tiff, K, Sc, dt, target_time):
    basefilename = os.path.splitext(in_tiff)[0]
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(os.path.join(BASE_DIR, in_tiff), in_asc)

    grid, tnld = init_simulation(in_asc, K, Sc, XYZunit)
    z_old = grid.at_node['topographic__elevation'].copy()
    soil_depth = grid.at_node['soil__depth']
    
    # Store the initial elevation and soil depth
    initial_elevation = z_old.copy()
    initial_soil_depth = soil_depth.copy()
    
    # Initialize total soil depth
    total_soil_depth = soil_depth.copy()

    # Initialize lists to store change in soil depth and total soil depth for each time step
    change_in_soil_depth_array = []
    total_soil_depth_array = []

    with rasterio.open(os.path.join(BASE_DIR, in_tiff)) as src:
        meta = src.meta.copy()

    asc_path = plot_save(grid, z_old, basefilename, 0, K, mean_res, XYZunit)

    num_steps = int(target_time / dt)
    for i in range(num_steps):
        tnld.run_one_step(dt)
        time = (i + 1) * dt
        
        # Calculate soil production for this time step
        production_rate = (pr / ps) * (P0 * np.exp(-total_soil_depth)) * dt
        z_new = grid.at_node['topographic__elevation']
        elevation_change = z_new - z_old

        erosion_exceeds = elevation_change > initial_soil_depth

        if np.any(erosion_exceeds):
            z_new[erosion_exceeds] = initial_elevation[erosion_exceeds] - initial_soil_depth[erosion_exceeds]
            print(f"Erosion exceeded soil depth at time {time} years. Adjusting elevation.")

        change_in_soil_depth = elevation_change + production_rate
        total_soil_depth += change_in_soil_depth
        grid.at_node['soil__depth'] = total_soil_depth
        z_old = z_new.copy()

        # Append the change in soil depth and total soil depth to the arrays
        change_in_soil_depth_array.append(change_in_soil_depth.copy())
        total_soil_depth_array.append(total_soil_depth.copy())

        # Output results every 500 years
        if time % 1000 == 0:
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

            # Save the current elevation as a GeoTIFF
            tiff_elevation_path = os.path.join(OUT_DIRtiff, f"{basefilename}_elevation_{time}yrs.tif")
            asc_to_tiff(asc_path, tiff_elevation_path, meta)
            
            # Save change in elevation and soil depth as PNGs
            grid_shape = grid.shape  # Get the shape of the grid
            plot_change(elevation_change, "Change in Elevation", basefilename, time, K, grid_shape)
            plot_change(change_in_soil_depth, "Change in Soil Depth", basefilename, time, K, grid_shape)
            plot_change(total_soil_depth, "Total Soil Depth", basefilename, time, K, grid_shape)

    in_prj = in_asc.replace('.asc', '.prj')
    os.remove(in_asc)
    os.remove(in_prj)
    print("Simulation completed. Temporary ASCII & PRJ files cleaned up.")

    # Convert lists to numpy arrays if needed
    change_in_soil_depth_array = np.array(change_in_soil_depth_array)
    total_soil_depth_array = np.array(total_soil_depth_array)

    print(f"Change in Soil Depth Array Shape: {change_in_soil_depth_array.shape}")
    print(f"Total Soil Depth Array Shape: {total_soil_depth_array.shape}")

    return change_in_soil_depth_array, total_soil_depth_array  # Return the arrays

#%%
# Example run parameters
K = 0.005  # Diffusion coefficient
Sc = 1.25  # Critical slope gradient
pr = 2000   # Ratio of production (example value)
ps = 1000   # Ratio of soil loss (example value)
P0 = 0.0003  # Initial soil production rate (example value, e.g., kg/m²/year)
h0 = 0.7   # Depth constant related to soil production (example value, e.g., meters)
dt = 50
target_time = 15000

run_simulation(INPUT_TIFF, K, Sc, dt, target_time)

# Call the simulation and capture the returned arrays
change_in_soil_depth, total_soil_depth = run_simulation(INPUT_TIFF, K, Sc, dt, target_time)

# Now you can see the arrays
print("Final Change in Soil Depth Array:")
print(change_in_soil_depth)

print("Final Total Soil Depth Array:")
print(total_soil_depth)

#%%
def plot_array(array, title):
    plt.figure(figsize=(10, 6))
    plt.plot(array, marker='o')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# After the simulation
plot_array(change_in_soil_depth, 'Change in Soil Depth Over Time')
plot_array(total_soil_depth, 'Total Soil Depth Over Time')


