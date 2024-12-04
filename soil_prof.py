# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:41:46 2023

@author: Selina Davila Olivera

This code creates soil profiles for SOC density given an excel file with 
previously calculated SOC density (kg/m3) values

This code can also be used to find sumulative sum of SOC for a soil profile
"""
#%%
## Import packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%
# read in excel file
path = "C:\\Users\\sdavilao\\OneDrive - University Of Oregon\\Desktop\\Hollow_Data\\SOC_data\\SOC_HCH.xlsx"

# make dataframe
SOC_df = pd.read_excel(path, names=['depth_m', 'upper_m', 'mid_m', 'sample', 'before_sieve_g', 
                                    'lt_2mm_fraction_g', 'gt_2mm_fraction_g', 'N', 'C', 'Fc', 'course_corr','soc_density_kgcm3'], 
                       skiprows=10, index_col= False)

#%%
# Specify the prefix
target_prefix = 'HCH2'
target_prefixA = 'HCH1A'
target_prefixB = 'HCH1B'

# Filter samples that start with the target prefix
HCH2_df = pd.DataFrame(SOC_df[SOC_df['sample'].str.startswith(target_prefix)])

HCH1A_df = pd.DataFrame(SOC_df[SOC_df['sample'].str.startswith(target_prefixA)])
HCH1B_df = pd.DataFrame(SOC_df[SOC_df['sample'].str.startswith(target_prefixB)])

# Calculate the cumulative sum for SOC
# HCH2_df['soc_cumsum'] = HCH2_df['soc_density_kgcm3'].cumsum()

#%%
## Plot SOC density profiles
plt.figure(figsize=(5, 7.5))
sns.scatterplot(x=HCH2_df.soc_density_kgcm3*1000, y=HCH2_df.mid_m, s = 80)

# Customize axis ticks and labels
plt.gca().invert_yaxis()
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')

# Add grid
plt.grid(True, linestyle='--', color='gray', alpha=0.5)

# Set labels
plt.ylabel('Depth (cm)', fontsize = 14)
plt.xlabel('SOC Density ($kg/m^3$)', fontsize = 14)

plt.show()

#%%
## Plot SOC density profiles

plt.figure(figsize=(5, 7.5))
sns.scatterplot(x=HCH1A_df.soc_density_kgcm3*1000, y=HCH1A_df.mid_m, marker='D', label='HCH1A - Hollow Body', s = 80)
sns.scatterplot(x=HCH1B_df.soc_density_kgcm3*1000, y=HCH1B_df.mid_m, marker='s', label="HCH1B - Hollow Scarp", s = 80)

# Customize axis ticks and labels
plt.gca().invert_yaxis()
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')

# Add grid
plt.grid(True, linestyle='--', color='gray', alpha=0.5)

# Set labels and legend
plt.ylabel('Depth (cm)', fontsize = 14)
plt.xlabel('SOC Density ($kg/m^3$)', fontsize = 14)
plt.legend(fontsize = 13, loc='upper right')

plt.show()

## Plot SOC totals with area under curve shaded
# plt.figure(figsize=(5,7.5))
# plt.fill_between(HCH2_df.soc_cumsum*1000, HCH2_df.mid_m, 375, color='skyblue', alpha=0.4)
# plt.plot(HCH2_df.soc_cumsum*1000, HCH2_df.mid_m)
# plt.gca().xaxis.set_ticks_position('top')
# plt.gca().xaxis.set_label_position('top')
# plt.gca().invert_yaxis()
# plt.ylabel('Depth (cm)')
# plt.xlabel('Total SOC Density with Depth ($kg/m^3$)')
