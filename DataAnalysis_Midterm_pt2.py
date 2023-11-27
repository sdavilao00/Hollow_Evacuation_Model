import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


pw = 1000  # kg/m3
ps = 1800  # kg/m3
g = 9.81  # m/s2
Sc = 1.25
slope = 40
slope_r = np.deg2rad(slope)
hollow = np.arctan(0.8 * np.tan(np.deg2rad(slope)))

## Cohesion of 2500 ##
# Define the number of Monte Carlo samples
num_samples = 100000

# Critical depth values
def crit_depth(y, z):
    denominator = ps * g * (np.cos(hollow) ** 2) * (
            np.tan(hollow) - ((1 - (z * (pw / ps))) * np.tan(y)))
    if denominator == 0:
        return float('inf')  # Handle division by zero
    return 2500 / denominator

# Recurrence interval, where variables depend on the results of critical depth
def ri(result1, w):
    B = (np.cos(hollow) ** (1/2)) * ((np.tan(slope_r)**2 - np.tan(hollow)**2)**(1/4)) * ((np.cos(hollow)**2 / np.cos(slope_r)**2 - 1)**(1/4))
    return (result1 ** 2) / (B ** 2 * (2 * w))

# Lists to store the results
results1 = []
results2 = []

# Perform the Monte Carlo simulation for critical depth
for _ in range(num_samples):
    #x = np.random.uniform(2000, 18000)
    y = np.random.uniform(np.deg2rad(33), np.deg2rad(40))
    z = np.random.uniform(0.4, 1)
    result1 = crit_depth(y, z)
    results1.append(result1)

# Perform the Monte Carlo simulation for RI based on the results of zcrit
for result1 in results1:
    w = np.random.uniform(0.001, 0.01)
    result2 = ri(result1, w)
    results2.append(result2)

# Calculate mean and standard deviation of RI
mean_result2 = np.mean(results2)
std_deviation_result2 = np.std(results2)

## Cohesion of 6800 ##
# Define the number of Monte Carlo samples
num_samples = 100000

# Critical depth values
def crit_depth(y, z):
    denominator = ps * g * (np.cos(hollow) ** 2) * (
            np.tan(hollow) - ((1 - (z * (pw / ps))) * np.tan(y)))
    if denominator == 0:
        return float('inf')  # Handle division by zero
    return 6800 / denominator

# Recurrence interval, where variables depend on the results of critical depth
def ri(result3, w):
    B = (np.cos(hollow) ** (1/2)) * ((np.tan(slope_r)**2 - np.tan(hollow)**2)**(1/4)) * ((np.cos(hollow)**2 / np.cos(slope_r)**2 - 1)**(1/4))
    return (result3 ** 2) / (B ** 2 * (2 * w))

# Lists to store the results
results3 = []
results4 = []

# Perform the Monte Carlo simulation for critical depth
for _ in range(num_samples):
    #x = np.random.uniform(2000, 18000)
    y = np.random.uniform(np.deg2rad(33), np.deg2rad(40))
    z = np.random.uniform(0.4, 1)
    result3 = crit_depth(y, z)
    results3.append(result3)

# Perform the Monte Carlo simulation for RI based on the results of zcrit
for result3 in results3:
    w = np.random.uniform(0.001, 0.01)
    result4 = ri(result3, w)
    results4.append(result4)

# Calculate mean and standard deviation of RI
mean_result4 = np.mean(results4)
std_deviation_result4 = np.std(results4)

## Cohesion of 11000 ##
# Define the number of Monte Carlo samples
num_samples = 100000

# Critical depth values
def crit_depth(y, z):
    denominator = ps * g * (np.cos(hollow) ** 2) * (
            np.tan(hollow) - ((1 - (z * (pw / ps))) * np.tan(y)))
    if denominator == 0:
        return float('inf')  # Handle division by zero
    return 11000 / denominator

# Recurrence interval, where variables depend on the results of critical depth
def ri(result5, w):
    B = (np.cos(hollow) ** (1/2)) * ((np.tan(slope_r)**2 - np.tan(hollow)**2)**(1/4)) * ((np.cos(hollow)**2 / np.cos(slope_r)**2 - 1)**(1/4))
    return (result5 ** 2) / (B ** 2 * (2 * w))

# Lists to store the results
results5 = []
results6 = []

# Perform the Monte Carlo simulation for critical depth
for _ in range(num_samples):
    #x = np.random.uniform(2000, 18000)
    y = np.random.uniform(np.deg2rad(33), np.deg2rad(40))
    z = np.random.uniform(0.4, 1)
    result5 = crit_depth(y, z)
    results5.append(result5)

# Perform the Monte Carlo simulation for RI based on the results of zcrit
for result5 in results5:
    w = np.random.uniform(0.001, 0.01)
    result6 = ri(result5, w)
    results6.append(result6)

# Calculate mean and standard deviation of RI
mean_result6 = np.mean(results6)
std_deviation_result6 = np.std(results6)


# Plot the histogram
plt.figure()
plt.hist(results2, bins=500, range = (0,50000), density=True, alpha=0.8, color='gray', edgecolor='k', label = 'C = 2500')
plt.hist(results4, bins=100, range = (0,50000), density=True, alpha=0.8, color='orange', edgecolor='k', label = 'C = 6800')
plt.hist(results6, bins=100, range = (0,50000), density=True, alpha=0.8, color='blue', edgecolor='k', label = 'C = 11000')
plt.legend()
plt.xlabel('Recurrence Interval (yrs)')
plt.ylabel('Probability Density')
plt.title('Monte Carlo Probability Distribution - slope = 40')
plt.xscale('log')

# # Annotate the plot with mean and standard deviation
# plt.annotate(f'Mean: {mean_result2:.2f}', xy=(0.3, 0.8), xycoords='axes fraction', fontsize=10, color='red')
# plt.annotate(f'Std Dev: {std_deviation_result2:.2f}', xy=(0.3, 0.75), xycoords='axes fraction', fontsize=10, color='red')
# plt.annotate(f'Mean: {mean_result4:.2f}', xy=(0.5, 0.3), xycoords='axes fraction', fontsize=10, color='red')
# plt.annotate(f'Std Dev: {std_deviation_result4:.2f}', xy=(0.5, 0.25), xycoords='axes fraction', fontsize=10, color='red')
# plt.annotate(f'Mean: {mean_result6:.2f}', xy=(0.7, 0.15), xycoords='axes fraction', fontsize=10, color='red')
# plt.annotate(f'Std Dev: {std_deviation_result6:.2f}', xy=(0.7, 0.1), xycoords='axes fraction', fontsize=10, color='red')

# Show the plot
plt.show()

# Create a DataFrame
df = pd.DataFrame(results2)

# Specify the file path where you want to save the CSV file
# csv_file_path = 'C:\\Users\\12092\\Documents\\Hollow_Evacuation_Data\\DataAnalysis\\MonteCarlo_40.csv'

# # Save the DataFrame as a CSV file
# df.to_csv(csv_file_path, index=False)  # Set index=False to omit writing row numbers

# print(f"Data saved to {csv_file_path}")
