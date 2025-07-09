# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:56:59 2025

@author: sdavilao
"""
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import glob
import imageio.v2 as imageio
import re
import gc
import pytopocomplexity
from IPython.display import display, Image
from osgeo import gdal
from landlab import imshowhs_grid
from landlab.components import TaylorNonLinearDiffuser, FlowAccumulator, StreamPowerEroder
from landlab.io import read_esri_ascii, write_esri_ascii

# Define input file name and directory
base_dir = os.path.join(os.getcwd()) # input file base directory
input_file = 'ugh16.tif'           # input file name
input_dir = os.path.join(base_dir, input_file)     # input file directory

# Check if base file directory exists. If not, create it
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    
def tiff_to_asc(in_path, out_path):
    with rasterio.open(in_path) as src:
        XYZunit = src.crs.linear_units   #assuming the unit of XYZ direction are the same
        mean_res = np.mean(src.res)
    
    gdal.UseExceptions()  # Enable exceptions for GDAL
    gdal.Translate(out_path, in_path, format='AAIGrid', xRes=mean_res, yRes=mean_res)
 
    print(f"The input GeoTIFF is temporarily converted to '{os.path.basename(out_path)}' with grid spacing {mean_res} ({XYZunit})")
    return mean_res, XYZunit
    
def asc_to_tiff(asc_path, tiff_path, meta):
    data = np.loadtxt(asc_path, skiprows=9)
    meta.update(dtype=rasterio.float32, count=1, compress='deflate')

    with rasterio.open(tiff_path, 'w', **meta) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    subfolder_path = os.path.dirname(tiff_path)
    last_two_folders = os.path.basename(os.path.dirname(subfolder_path)) + '/' + os.path.basename(subfolder_path)
    print(f"'{os.path.basename(tiff_path)}' is saved under subfolder '{last_two_folders}'")
    
def init_landlab_components(asc_file, XYZunit=None, Hdiffusion=True, SPincision=True, D=None, Sc=None, K=None, m=0.5, n=1.0, omega=0, U=0, NumTaylorTerm=2):
    # Define unit conversion factors
    ft2mUS = 1200/3937   #US survey foot to meter conversion factor 
    ft2mInt = 0.3048     #International foot to meter conversion factor 

    # Read the ASCII file into a Landlab grid
    grid, _ = read_esri_ascii(asc_file, name='topographic__elevation') #the xy grid spacing must be equal
    # Set boundary conditions: open boundaries allow sediment to move in/out of the study area
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)  #(right, top, left, bottom) bounaries

    # Check the unit of XYZ and make unit conversion of D and K when needed
    if XYZunit is None:
        print("The function assumes the input XYZ units are in meters.")
    elif XYZunit is not None:
        if any(unit in XYZunit.lower() for unit in ["metre".lower(), "meter".lower()]):
            print("Input XYZ units are in meters. No unit conversion is made")
        elif any(unit in XYZunit.lower() for unit in ["foot".lower(), "feet".lower(), "ft".lower()]):  
            if any(unit in XYZunit.lower() for unit in ["US".lower(), "United States".lower()]):
                print("Input XYZ units: US survey feet. Unit conversion to meters is applied during the simulation.")
                if D is not None:
                    D = D / (ft2mInt ** 2)    # Convert D from m^2/yr to ft^2/yr
                if K is not None:
                    K = K / (ft2mInt ** 0.5)  # Convert K from m^0.5/yr to ft^0.5/yr
            else:
                print("Input XYZ units: international feet. Unit conversion to meters is applied during the simulation.")
                if D is not None:
                    D = D / (ft2mInt ** 2)    # Convert D from m^2/yr to ft^2/yr
                if K is not None:
                    K = K / (ft2mInt ** 0.5)  # Convert K from m^0.5/yr to ft^0.5/yr
        else:
            message = (
            "WARNING: The code execution is stopped."
            "The input XYZ units must be in feet or meters."
            )
            raise RuntimeError(message)
        
    # Initiate Landlab components 
    TNLD = None
    SP = None
    fa = None

    if Hdiffusion:
        if D is None or Sc is None:
            raise ValueError("D and Sc must be provided when Hdiffusion is True")
        # For nonlinear hillslope diffusion
        TNLD = TaylorNonLinearDiffuser(grid, linear_diffusivity=D, slope_crit=Sc, dynamic_dt=True, nterms=NumTaylorTerm, if_unstable = "pass")
    
    if SPincision:
        if K is None:
            raise ValueError("K must be provided when SPincision is True")
        # For flow routing
        fa = FlowAccumulator(grid, flow_director='D8')
        # For stream power erosion
        SP = StreamPowerEroder(grid, K_sp=K, m_sp=m, n_sp=n, threshold_sp=omega)
    
    return grid, TNLD, SP, fa, U

def run_landlab_simulation(in_tiff, dt, end_time, U=0, Hdiffusion=None, D=None, Sc=None, NumTaylorTerm=2, SPincision=None, K=None, m=0.5, n=1.0, omega=0, savetiff=True, savepng=True, savegif=True):
    if Hdiffusion is None or SPincision is None:
        raise ValueError("Both Hdiffusion and SPincision must be explicitly set to True or False")
    if not (savetiff or savepng):
        raise ValueError("At least one of savetiff or savepng must be True")
    if savegif and not savepng:
        raise ValueError("When savegif is True, savepng must also be True")
    if not (Hdiffusion or SPincision):
        raise ValueError("At least one of Hdiffusion or SPincision must be True")

    BASE_DIR = os.path.dirname(in_tiff)
    basefilename = os.path.splitext(os.path.basename(in_tiff))[0]
    in_asc = os.path.join(BASE_DIR, f"{basefilename}.asc")
    mean_res, XYZunit = tiff_to_asc(in_tiff, in_asc) #convert input GeoTIFF to ASCII, and determine the XYZ units

    grid, tnld, sp, fa, U = init_landlab_components(in_asc, XYZunit, Hdiffusion, SPincision, D, Sc, K, m, n, omega, U, NumTaylorTerm)
    z = grid.at_node['topographic__elevation']

    with rasterio.open(in_tiff) as src:
        meta = src.meta.copy()

    # Setup output directories
    if Hdiffusion and SPincision:
        subfolder = 'simulation_HDSP'
    elif Hdiffusion:
        subfolder = 'simulation_HD'
    elif SPincision:
        subfolder = 'simulation_SP'
    OUT_DIR = os.path.join(BASE_DIR, subfolder)
    os.makedirs(OUT_DIR, exist_ok=True)
    if savepng:
        OUT_DIRpng = os.path.join(OUT_DIR, 'PNGs')
        os.makedirs(OUT_DIRpng, exist_ok=True)
    if savetiff:
        OUT_DIRtiff = os.path.join(OUT_DIR, 'GeoTIFFs')
        os.makedirs(OUT_DIRtiff, exist_ok=True)
    if savegif:
        OUT_DIRgif = os.path.join(OUT_DIR, 'GIFs')
        os.makedirs(OUT_DIRgif, exist_ok=True)

    png_files = []
    num_steps = int(end_time / dt)
    for i in range(num_steps + 1):             # +1 to include initial state
        time = i * dt
        if i > 0:
            small_dt = max(dt / 10, 1)         # Ensure the small timestep is at least 1 year
            for _ in range(int(dt / small_dt)):
                if U != 0:
                    z[grid.core_nodes] += U * small_dt
                if SPincision:
                    fa.run_one_step()              # Update flow routing
                    sp.run_one_step(small_dt)      # Run stream power erosion
                if Hdiffusion:
                    tnld.run_one_step(small_dt)    # Run nonlinear hillslope diffusion

        if savepng or savetiff:
            plt.figure(figsize=(6, 5.25))
            imshowhs_grid(grid, z, plot_type="Hillshade")
            title = f"{basefilename} Time: {time} years (U = {U} m/yr)\n"
            if Hdiffusion:
                title += f"Nonlinear HillslopeDiffusion (D = {D} m$^{{2}}$/yr)"
            if Hdiffusion and SPincision:
                title += f"\n + "
            if SPincision:
                title += f"Stream Power Incision (K = {K} m$^{{{1-m}}}$/yr)"
            plt.title(title, fontsize='small', fontweight="bold")
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
            
            # Add uplift rate text at the lower left
            plt.text(0.01, 0.01, f'U = {U:.2e} m/yr', transform=plt.gca().transAxes, fontsize='x-small', verticalalignment='bottom')
            
            if savepng:
                png_path = os.path.join(OUT_DIRpng, f"{basefilename}_{time}yrs.png")
                plt.savefig(png_path, dpi=150)
                png_files.append(png_path)
                print(f"'{os.path.basename(png_path)}' is saved")
            
            plt.close()

            if savetiff:
                asc_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs.asc")
                write_esri_ascii(asc_path, grid, names=['topographic__elevation'], clobber=True)
                
                tiff_path = os.path.join(OUT_DIRtiff, f"{basefilename}_{time}yrs.tif")
                asc_to_tiff(asc_path, tiff_path, meta)
                
                os.remove(asc_path)

    if savegif and png_files:
        # Find all PNG files for the simulation
        pattern = f"{basefilename}_*.png"
        png_files = glob.glob(os.path.join(OUT_DIRpng, pattern))
        
        # Sort files based on the year in the filename
        png_files.sort(key=lambda x: int(re.search(r'_(\d+)yrs\.png$', x).group(1)))
        
        # Read all images
        images = [imageio.imread(f) for f in png_files]
        
        # Save the gif
        output_file = os.path.join(OUT_DIRgif, f'{basefilename}_0-{end_time}yrs_animation.gif')
        imageio.mimsave(output_file, images, 'GIF', duration=2.0, loop=0)
        
        print(f"'{os.path.basename(output_file)}' is saved")
        
        # Display the GIF
        display(Image(filename=output_file))

    in_prj = in_asc.replace('.asc', '.prj')
    in_aux = in_asc.replace('.asc', '.asc.aux.xml')
    os.remove(in_asc)
    os.remove(in_prj)
    os.remove(in_aux)
    print("Simulation completed. Temporary ASCII & PRJ files cleaned up.")
    
    
def process_single_file(tif_file, output_dir, pytopocomplexity_method, spatial_scale, savetiff, savepng, colormax1=None, colormax2=None):
    filename = os.path.basename(tif_file)
    method_name = pytopocomplexity_method.__name__

    print(f"Processing {filename}")

    if savetiff:
        out_dir_tif = os.path.join(output_dir, f"GeoTIFFs_{method_name}")
        os.makedirs(out_dir_tif, exist_ok=True)
    if savepng:
        out_dir_png = os.path.join(output_dir, f"PNGs_{method_name}")
        os.makedirs(out_dir_png, exist_ok=True)
    
    # Extract the time from the filename
    time_match = re.search(r'_(\d+)yrs', tif_file)
    if time_match and time_match.group(1) == '0':
        if method_name == 'CWTMexHat':
            analyzer = pytopocomplexity_method(Lambda=spatial_scale)
            _, cwtresult = analyzer.analyze(tif_file, conv_method='fft', chunk_processing=True, chunksize=(512,512))
            colormax1 = round(np.nanpercentile(cwtresult, 99),2)
        elif method_name == 'FracD':
            analyzer = pytopocomplexity_method(window_size=spatial_scale)
            _, _, SE_result, _, _ = analyzer.analyze(tif_file, variograms=False, chunk_processing=True, chunksize=(512, 512))
            colormax1 = round(np.nanpercentile(SE_result, 99),2)
        elif method_name == 'RugosityIndex':
            analyzer = pytopocomplexity_method(window_size=spatial_scale)
            _, rugosityresult, _ = analyzer.analyze(tif_file, slope_correction=True, chunk_processing=True, chunksize=(512, 512))
            colormax1 = round(np.nanpercentile(rugosityresult, 99),2) 
        elif method_name == 'TPI':
            analyzer = pytopocomplexity_method(window_size=spatial_scale)
            _, TPI_result, TPIabs_result, _ = analyzer.analyze(tif_file, chunk_processing=True, chunksize=(512, 512))
            colormax1 = round(np.max([np.abs(np.nanpercentile(TPI_result, 1)), np.abs(np.nanpercentile(TPI_result, 99))]),2)
            colormax2 = round(np.nanpercentile(TPIabs_result, 99),2)
    else:
        if method_name == 'CWTMexHat':
            analyzer = pytopocomplexity_method(Lambda=spatial_scale)
            analyzer.analyze(tif_file, conv_method='fft', chunk_processing=True, chunksize=(512,512))
        elif method_name == 'FracD':
            analyzer = pytopocomplexity_method(window_size=spatial_scale)
            analyzer.analyze(tif_file, variograms=False, chunk_processing=True, chunksize=(512, 512))
        elif method_name == 'RugosityIndex':
            analyzer = pytopocomplexity_method(window_size=spatial_scale)
            analyzer.analyze(tif_file, slope_correction=True, chunk_processing=True, chunksize=(512, 512))
        elif method_name == 'TPI':
            analyzer = pytopocomplexity_method(window_size=spatial_scale)
            analyzer.analyze(tif_file, chunk_processing=True, chunksize=(512, 512))
    
    output_file = os.path.splitext(filename)[0] + f'_{method_name}({spatial_scale}m)'
    if savetiff:
        output_tif = os.path.join(out_dir_tif, output_file + '.tif')
        analyzer.export_result(output_tif)

    if savepng:
        output_png = os.path.join(out_dir_png, output_file + '.png')
        if method_name == 'CWTMexHat':
            analyzer.plot_result(savefig=True, output_dir=out_dir_png, figshow=False, showhillshade=True, cwtcolormax=colormax1)
        elif method_name == 'FracD': 
            analyzer.plot_result(savefig=True, output_dir=out_dir_png, figshow=False, showhillshade=True, showse=True, showr2=True, secolormax=colormax1)
        elif method_name == 'RugosityIndex':
            analyzer.plot_result(savefig=True, output_dir=out_dir_png, figshow=False, showhillshade=True, rugositycolormax=colormax1)
        elif method_name == 'TPI':
            analyzer.plot_result(savefig=True, output_dir=out_dir_png, figshow=False, showhillshade=True, showtpi=True, showabstpi=True, tpicolormax=colormax1, abstpicolormax=colormax2)
    
    gc.collect()
    
    return colormax1, colormax2

def run_pytopocomplexity(input_dir, output_dir, method, spatial_scale, savetiff=True, savepng=True, savegif=True):
    method_map = {
        'CWTMexHat': pytopocomplexity.CWTMexHat,
        'FracD': pytopocomplexity.FracD,
        'RugosityIndex': pytopocomplexity.RugosityIndex,
        'TPI': pytopocomplexity.TPI
    }
    
    if method == 'CWTMexHat':
        print(f"Running {method} from pyTopoComplexity with Lambda = {spatial_scale} m")
    else:
        print(f"Running {method} from pyTopoComplexity with window size = {spatial_scale} grids")

    if method not in method_map:
        raise ValueError("Invalid method. Please use 'CWTMexHat', 'FracD', 'RugosityIndex', or 'TPI'.")
    
    if not (savetiff or savepng):
        raise ValueError("At least one of savetiff or savepng must be True")
    
    if savegif and not savepng:
        raise ValueError("When savegif is True, savepng must also be True")
    
    method_class = method_map[method]
    tif_files = glob.glob(os.path.join(input_dir, '*.tif'))
    
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")
    
    # Sort tif_files based on the year in the filename, starting with the smallest
    tif_files.sort(key=lambda x: int(re.search(r'_(\d+)yrs', x).group(1)))
    
    # Process the tif files
    colormax1, colormax2 = None, None
    for tif_file in tif_files:
        colormax1, colormax2 = process_single_file(tif_file, output_dir, method_class, spatial_scale, savetiff, savepng, colormax1, colormax2)
    
    if savegif:
        png_dir = os.path.join(output_dir, f"PNGs_{method}")
        gif_dir = os.path.join(output_dir, f"GIFs_{method}")
        os.makedirs(gif_dir, exist_ok=True)
        
        png_files = sorted(glob.glob(os.path.join(png_dir, '*.png')))
        
        if png_files:
            # Sort png_files based on the year in the filename, starting with the smallest
            png_files.sort(key=lambda x: int(re.search(r'_(\d+)yrs_', x).group(1)))
            
            images = []
            max_year = 0
            basefilename = ''
            for f in png_files:
                img = imageio.imread(f)
                images.append(img)
                year = int(re.search(r'_(\d+)yrs_', f).group(1))
                if year > max_year:
                    max_year = year
                if not basefilename:
                    basefilename = os.path.basename(f).split('_')[0]
            
            if images:
                # Find the maximum dimensions
                max_height = max(img.shape[0] for img in images)
                max_width = max(img.shape[1] for img in images)
                
                # Pad images to the same size, aligning to the upper left
                padded_images = []
                for img in images:
                    h, w = img.shape[:2]
                    pad_bottom = max_height - h
                    pad_right = max_width - w
                    padded_img = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant')
                    padded_images.append(padded_img)
                
                gif_path = os.path.join(gif_dir, f'{basefilename}_0-{max_year}yrs_animation.gif')
                imageio.mimsave(gif_path, padded_images, duration=2.0, loop=0)
                print(f"'{os.path.basename(gif_path)}' is saved")
                display(Image(filename=gif_path))
            else:
                print("No images found to create GIF animation.")
        else:
            print("No PNG files found to create GIF animation.")
            
#%%
# Run the landlab simulation with only hillslope diffusion (reproduce Fig. 6a-6e in Booth et al., 2017)
run_landlab_simulation(
    input_dir,
    dt=50,            # Time step size (years)
    end_time=150,     # Final simulation time (years)
    U=0,                # Uplift rate, m/y   
    Hdiffusion=True,    # Enable/Disable hillslope diffusion
    D=0.0029,           # Hillslope diffusion coefficient, m^2/y (0.0029 is used in Booth et al., 2017)
    Sc=1.25,            # Critical gradient for hillslope diffusion, m/m (1.25 is used in Booth et al., 2017)
    NumTaylorTerm=2,    # Number of terms in the Taylor expansion
    SPincision=False,   # Enable/Disable stream power incision
    savetiff=True,      # Save GeoTIFF files
    savepng=True,      # Save PNG files
    savegif=False       # Save GIF file
)