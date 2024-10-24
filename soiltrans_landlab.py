# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:28:49 2024

@author: sdavilao
"""
#%%
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Image
from osgeo import gdal
from landlab import imshowhs_grid
from landlab.components import TaylorNonLinearDiffuser
from landlab.io import read_esri_ascii, write_esri_ascii
#%%
# Input geotiff file and directory
BASE_DIR = os.path.join(os.getcwd(), 'ExampleDEM')
INPUT_TIFF = 'hc_smooth.tif'

# Setup output directory
OUT_DIR = os.path.join(BASE_DIR, 'simulation_results')
OUT_DIRpng = os.path.join(OUT_DIR, 'PNGs')
OUT_DIRtiff = os.path.join(OUT_DIR, 'GeoTIFFs')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIRpng, exist_ok=True)
os.makedirs(OUT_DIRtiff, exist_ok=True)
#%%
def tiff_to_asc(in_path, out_path):
    with rasterio.open(in_path) as src:
        XYZunit = src.crs.linear_units   #assuming the unit of XYZ direction are the same
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
ft2mUS = 1200/3937   #US survey foot to meter conversion factor 
ft2mInt = 0.3048     #International foot to meter conversion factor 

def init_simulation(asc_file, K, Sc, XYZunit=None):
    grid, _ = read_esri_ascii(asc_file, name='topographic__elevation') #the xy grid spacing must be equal
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False) #boundaries open: allowing sediment move in/out of the study area

    # Check the unit of XYZ and make unit conversion of K when needed
    if XYZunit is None:
        print("The function assumes the input XYZ units are in meters.")
        TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=K, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable = "pass")
    elif XYZunit is not None:
        if any(unit in XYZunit.lower() for unit in ["metre".lower(), "meter".lower()]):
            print("Input XYZ units are in meters. No unit conversion is made")
            TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=K, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable = "pass")
        elif any(unit in XYZunit.lower() for unit in ["foot".lower(), "feet".lower(), "ft".lower()]):  
            if any(unit in XYZunit.lower() for unit in ["US".lower(), "United States".lower()]):
                print("Input XYZ units are in US survey feet. A unit conversion to meters is made for K")
                Kc = K / (ft2mUS ** 2)
                TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=Kc, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable = "pass")
            else:
                print("Input XYZ units are in international feet. A unit conversion to meters is made for K")
                Kc = K / (ft2mInt ** 2)
                TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=Kc, slope_crit=Sc, dynamic_dt=True, nterms=2, if_unstable = "pass")
        else:
            message = (
            "WARNING: The code excution is stopped. "
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
        plt.xlabel(f'X-axis grids \n(grid size ≈ {1.0} [meters])', fontsize = 'small')
        plt.ylabel(f'Y-axis grids \n(grid size ≈ {1.0} [meters])', fontsize = 'small') 
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(OUT_DIRpng, f"{basefilename}_{time}yrs_(K={K}).png"), dpi=150)
    plt.close()
    print(f"'{basefilename}_{time}yrs_(K={K}).png' saved to 'simulation_results' folder")

    asc_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}_(K={K})yrs.asc")
    write_esri_ascii(asc_path, grid, names=['topographic__elevation'], clobber=True)
    
    return asc_path
#%% ORIGINAL SOIL TRANSPORT FROM LARRY

# def run_simulation(in_tiff, K, Sc, dt, target_time):
#     basefilename = os.path.splitext(in_tiff)[0]
#     in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
#     mean_res, XYZunit = tiff_to_asc(os.path.join(BASE_DIR, in_tiff), in_asc) #convert input GeoTIFF to ASCII, and determine the XYZ units

#     grid, tnld = init_simulation(in_asc, K, Sc, XYZunit)
#     z = grid.at_node['topographic__elevation']

#     with rasterio.open(os.path.join(BASE_DIR, in_tiff)) as src:
#         meta = src.meta.copy()

#     asc_path = plot_save(grid, z, basefilename, 0, K, mean_res, XYZunit)
#     os.remove(asc_path)

#     num_steps = int(target_time / dt)
#     for i in range(num_steps):
#         tnld.run_one_step(dt)
#         time = (i + 1) * dt
#         asc_path = plot_save(grid, z, basefilename, time, K, mean_res, XYZunit)
        
#         tiff_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs_(K={K}).tif")
#         asc_to_tiff(asc_path, tiff_path, meta)
        
#         os.remove(asc_path)
    
        
#     in_prj = in_asc.replace('.asc', '.prj')
#     os.remove(in_asc)
#     os.remove(in_prj)
# #     print("Simulation completed. Temporary ASCII & PRJ files cleaned up.")

#%%  


# Constants for soil production
pr = 2000  # Rock density (kg/m^3)
ps = 1000  # Soil density (kg/m^3)
P0 = 0.003  # Initial production rate (m/year)
h0 = 0.4  # Decay constant (meters)


def initialize_soil_depth_grid(tiff_path, initial_soil_depth=0.5):
    """
    Create a grid of the same size as the input TIFF file and set the soil depth everywhere to the initial value.

    :param tiff_path: Path to the input GeoTIFF file.
    :param initial_soil_depth: The initial soil depth value (in meters).
    :return: A numpy array representing the soil depth grid.
    """
    with rasterio.open(tiff_path) as src:
        # Get the shape of the input DEM (number of rows and columns)
        height, width = src.shape
    
    # Create a new grid with the same shape, setting the soil depth to 0.5 meters everywhere
    soil_depth_grid = np.full((height, width), initial_soil_depth)
    
    return soil_depth_grid

def calculate_soil_production(soil_depth_grid, dt):
    """
    Calculate the soil production for each grid cell using the provided formula.
    
    :param soil_depth_grid: The current soil depth at each grid cell.
    :param dt: Time step duration (in years).
    :return: A grid of soil production values to be added to the current soil depth.
    """
    # Soil production equation: (pr/ps) * (P0 * e^(-h/h0)) * dt
    production_rate = (pr / ps) * (P0 * np.exp(-soil_depth_grid / h0)) * dt
    return production_rate

def save_elevation_as_png(elevation_grid, step):
    """
    Save the elevation grid as a PNG image.

    :param elevation_grid: The 2D numpy array representing elevation values.
    :param step: Current step of the simulation, used to name the file.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(elevation_grid, cmap='terrain', origin='upper')
    plt.colorbar(label='Elevation (m)')
    plt.title(f'Elevation at Step {step * 1000} Years')
    plt.xlabel('X')
    plt.ylabel('Y')

    output_png_path = os.path.join(OUT_DIRpng, f'elevation_step_{step}.png')
    plt.savefig(output_png_path, dpi=150)
    plt.close()
    
    print(f"Elevation PNG saved as '{output_png_path}'")

def run_simulation(in_tiff, K, Sc, dt, target_time):
    """
    Run the soil transport simulation using the original code logic.
    """
    basefilename = os.path.splitext(in_tiff)[0]
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(os.path.join(BASE_DIR, in_tiff), in_asc) # Convert input GeoTIFF to ASCII, and determine the XYZ units

    grid, tnld = init_simulation(in_asc, K, Sc, XYZunit)
    z = grid.at_node['topographic__elevation']
    grid_shape = (grid.number_of_node_rows, grid.number_of_node_columns)  # Ensure we know the 2D shape

    with rasterio.open(os.path.join(BASE_DIR, in_tiff)) as src:
        meta = src.meta.copy()

    # Initialize soil depth grid
    soil_depth_grid = initialize_soil_depth_grid(in_tiff)

    # Save initial state
    asc_path = plot_save(grid, z, basefilename, 0, K, mean_res, XYZunit)
    os.remove(asc_path)

    # Simulation parameters
    num_steps = int(target_time / dt)
    previous_z = z.copy().reshape(grid_shape)

    for i in range(num_steps):
        # Step 1: Run soil transport on the current elevation grid
        tnld.run_one_step(dt)
        time = (i + 1) * dt

        # Reshape z to 2D for consistency
        current_z = z.reshape(grid_shape)

        # Step 2: Calculate the difference between the new and previous elevation grids
        elevation_difference = previous_z - current_z

        # Step 3: Use the soil depth grid to calculate soil production
        soil_production = calculate_soil_production(soil_depth_grid, dt)

        # Step 4: Add the soil production to the elevation difference grid
        elevation_difference += soil_production

        # Step 5: Update the elevation grid by applying the soil production to the new elevation
        current_z += elevation_difference

        # Flatten to 1D to update the grid object
        z[:] = current_z.flatten()

        # Save the updated elevation as a PNG for this time step
        save_elevation_as_png(current_z, i + 1)

        # Save the updated GeoTIFF file
        asc_path = plot_save(grid, z, basefilename, time, K, mean_res, XYZunit)
        tiff_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs_(K={K}).tif")
        asc_to_tiff(asc_path, tiff_path, meta)
        os.remove(asc_path)

        # Update the previous elevation grid for the next time step
        previous_z = current_z.copy()

    in_prj = in_asc.replace('.asc', '.prj')
    os.remove(in_asc)
    os.remove(in_prj)
    print("Simulation completed. Temporary ASCII & PRJ files cleaned up.")

# Example of running the simulation
K = 0.005  # Diffusion coefficient
Sc = 1.25  # Critical slope gradient
dt = 1000  # Time step duration (in years)
total_time = 15000  # Total simulation time (in years)

run_simulation(INPUT_TIFF, K, Sc, dt, total_time)

#%%  
if __name__ == "__main__":
    # diffusion coefficient, m^2/y
    #K = 0.011
    #K = 0.0056
    K = 0.004     #Used in Fig.6 of Booth et al. (2017)
    #K = 0.0015
    #K = 0.00082

    Sc = 1.25   # critical slope gradient, m/m
    dt = 1000   # time step size (years)
    end_time = 15000  # final simulation time (years)

    run_simulation(INPUT_TIFF, K, Sc, dt, end_time)

#%%

def display_image(time):
    filename = f"{os.path.splitext(INPUT_TIFF)[0]}_{time}yrs_(K={K}).png"
    filepath = os.path.join(OUT_DIRpng, filename)
    if os.path.exists(filepath):
        display(Image(filename=filepath))
    else:
        print("No file found for this year.")

print(f"Diffusion coefficient K = {K} m²/yr")

# Create a slider of time (dt)
slider = widgets.IntSlider(
    value=0,
    min=0,
    max=end_time,  # end_time should be defined in your earlier cells
    step=dt,  # dt should be defined as your timestep size in years
    description='Time (years):', #Name of the slider
    continuous_update=False
)

# Interactive display of the image
widgets.interactive(display_image, time=slider)

#%%

# import numpy as np
# import rasterio
# import os
# import matplotlib.pyplot as plt

# # Constants for soil production
# pr = 2000  # Rock density (kg/m^3)
# ps = 1000  # Soil density (kg/m^3)
# P0 = 0.003  # Initial production rate (m/year)
# h0 = 0.4  # Decay constant (meters)

# # Assuming BASE_DIR and INPUT_TIFF are already defined as in your original code
# BASE_DIR = os.path.join(os.getcwd(), 'ExampleDEM')
# INPUT_TIFF = 'hc_clip.tif'

# # Directory to save PNGs
# OUT_DIRpng = os.path.join(BASE_DIR, 'simulation_results', 'PNGs')
# os.makedirs(OUT_DIRpng, exist_ok=True)

# def initialize_soil_depth_grid(tiff_path, initial_soil_depth=0.5):
#     """
#     Create a grid of the same size as the input TIFF file and set the soil depth everywhere to the initial value.

#     :param tiff_path: Path to the input GeoTIFF file.
#     :param initial_soil_depth: The initial soil depth value (in meters).
#     :return: A numpy array representing the soil depth grid.
#     """
#     with rasterio.open(tiff_path) as src:
#         # Get the shape of the input DEM (number of rows and columns)
#         height, width = src.shape
    
#     # Create a new grid with the same shape, setting the soil depth to 0.5 meters everywhere
#     soil_depth_grid = np.full((height, width), initial_soil_depth)
    
#     return soil_depth_grid

# def calculate_soil_production(soil_depth_grid, dt):
#     """
#     Calculate the soil production for each grid cell using the provided formula.
    
#     :param soil_depth_grid: The current soil depth at each grid cell.
#     :param dt: Time step duration (in years).
#     :return: A grid of soil production values to be added to the current soil depth.
#     """
#     # Soil production equation: (pr/ps) * (P0 * e^(-h/h0)) * dt
#     production_rate = (pr / ps) * (P0 * np.exp(-soil_depth_grid / h0)) * dt
#     return production_rate

# def save_soil_depth_as_png(soil_depth_grid, step):
#     """
#     Save the soil depth grid as a PNG image.

#     :param soil_depth_grid: The 2D numpy array representing soil depth values.
#     :param step: Current step of the simulation, used to name the file.
#     """
#     plt.figure(figsize=(8, 6))
#     plt.imshow(soil_depth_grid, cmap='terrain', origin='upper')
#     plt.colorbar(label='Soil Depth (m)')
#     plt.title(f'Soil Depth at Step {step * 1000} Years')
#     plt.xlabel('X')
#     plt.ylabel('Y')

#     output_png_path = os.path.join(OUT_DIRpng, f'soil_depth_step_{step}.png')
#     plt.savefig(output_png_path, dpi=150)
#     plt.close()
    
#     print(f"Soil depth PNG saved as '{output_png_path}'")

# # Use the same TIFF path from the start of the original code
# tiff_path = os.path.join(BASE_DIR, INPUT_TIFF)

# # Initialize soil depth grid
# soil_depth_grid = initialize_soil_depth_grid(tiff_path)

# # Simulation parameters
# dt = 1000  # Time step duration (in years)
# total_time = 15000  # Total simulation time (in years)
# num_steps = int(total_time / dt)

# # Run the simulation, adding soil production at each time step and saving PNGs
# for step in range(num_steps):
#     # Calculate soil production for this time step
#     soil_production = calculate_soil_production(soil_depth_grid, dt)
    
#     # Add the soil production to the current soil depth
#     soil_depth_grid += soil_production
    
#     # Save the soil depth as a PNG for this time step
#     save_soil_depth_as_png(soil_depth_grid, step + 1)

# print("Simulation completed.")








































