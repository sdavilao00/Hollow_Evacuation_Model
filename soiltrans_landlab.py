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
import cProfile
#%%
# Input geotiff file and directory
BASE_DIR = os.path.join(os.getcwd())
INPUT_TIFF = 'ext4.tif'

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

def run_simulation(in_tiff, K, Sc, dt, target_time):
    basefilename = os.path.splitext(in_tiff)[0]
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(os.path.join(BASE_DIR, in_tiff), in_asc) #convert input GeoTIFF to ASCII, and determine the XYZ units

    grid, tnld = init_simulation(in_asc, K, Sc, XYZunit)
    z = grid.at_node['topographic__elevation']

    with rasterio.open(os.path.join(BASE_DIR, in_tiff)) as src:
        meta = src.meta.copy()

    asc_path = plot_save(grid, z, basefilename, 0, K, mean_res, XYZunit)
    os.remove(asc_path)

    num_steps = int(target_time / dt)
    for i in range(num_steps):
        tnld.run_one_step(dt)
        time = (i + 1) * dt
        asc_path = plot_save(grid, z, basefilename, time, K, mean_res, XYZunit)
        
        tiff_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs_(K={K}).tif")
        asc_to_tiff(asc_path, tiff_path, meta)
        
        os.remove(asc_path)
    
        
    in_prj = in_asc.replace('.asc', '.prj')
    os.remove(in_asc)
    os.remove(in_prj)
    print("Simulation completed. Temporary ASCII & PRJ files cleaned up.")

#%%
# Example run parameters
K = 0.005  # Diffusion coefficient
Sc = 1.25  # Critical slope gradient
dt = 50  # Time step (years)
total_time = 150  # Total simulation time (years)

run_simulation(INPUT_TIFF, K, Sc, dt, total_time)
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
    max=total_time,  # end_time should be defined in your earlier cells
    step=dt,  # dt should be defined as your timestep size in years
    description='Time (years):', #Name of the slider
    continuous_update=False
)

# Interactive display of the image
widgets.interactive(display_image, time=slider)
























