# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:41:46 2023

@author: Selina Davila Olivera
"""

## Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## read in csv file
path = "C:\\Users\\12092\\Documents\\Hollow_Data\\SOC_data\\SOC_HCH.xlsx"

## make dataframe
SOC_df = pd.read_excel(path, names=['depth_m', 'upper_m', 'mid_m', 'sample', 'before_sieve_g', 
                                    'lt_2mm_fraction_g', 'gt_2mm_fraction_g', 'N', 'C', 'Fc', 'course_corr','soc_density_kgcm3'], 
                       skiprows=12)

## Specify the prefix
target_prefix = 'HCH2'

## Filter samples that start with the target prefix
HCH2_df = pd.DataFrame(SOC_df[SOC_df['sample'].str.startswith(target_prefix)])

## Calculate the cumulative sum for SOC
HCH2_df['soc_cumsum'] = HCH2_df['soc_density_kgcm3'].cumsum()

## Plot SOC profiles
plt.figure(figsize=(5,7.5))
plt.scatter(HCH2_df.soc_density_kgcm3*1000, HCH2_df.mid_m)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', color='gray', alpha=0.5, zorder=1)
plt.ylabel('Depth (m)')
plt.xlabel('SOC Stock ($kg/m^3$)')

## Plot SOC totals with area under curve shaded
plt.figure(figsize=(5,7.5))
plt.fill_between(HCH2_df.soc_cumsum*1000, HCH2_df.mid_m, 375, color='skyblue', alpha=0.4)
plt.plot(HCH2_df.soc_cumsum*1000, HCH2_df.mid_m)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.gca().invert_yaxis()
plt.ylabel('Depth (m)')
plt.xlabel('Overall SOC Stock ($kg/m^3$)')
