import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns 

# Given X values (independent variable)
X_values = [
    0.872437, 0.841073, 0.881849, 0.932277, 0.85545, 0.421472, 0.698179, 0.905916,
    1.00465, 0.85605, 0.744414, 0.960037, 0.651169, 0.890709, 0.454403, 0.301405,
    0.821296, 0.913297, 0.733394, 0.8511, 0.82394, 0.832595, 0.895107, 0.795758,
    0.904717, 0.891266, 0.784121, 0.902367, 0.845243, 0.846895, 0.838657, 0.917807,
    0.834244, 0.850871, 0.622304, 0.811978, 0.788007, 0.692322, 0.817353, 0.807673,
    0.678594, 0.635248, 0.814119, 0.775922, 0.7521, 0.808502, 0.861484, 0.863621,
    0.779649, 0.883719, 0.729697, 0.806591, 0.920318, 0.97026, 0.882973, 0.898598,
    0.847474, 0.938142, 0.952514, 0.647407, 0.706678, 0.709695, 0.400091, 0.389297,
    0.445974, 0.344344, 0.813336, 0.755817, 0.269651, 0.363154, 0.273894, 0.302246,
    0.211429, 0.426045, 0.421966, 0.80792, 0.85037, 0.697557, 0.543794, 0.715885,
    0.81153, 0.764267, 0.831186, 0.891238, 0.371101, 0.694903, 0.826924, 0.962194,
    0.380399, 0.320821, 0.855277, 0.319772, 0.937661, 0.883361, 0.794429, 0.785346,
    0.596261, 0.732403, 0.701114, 0.383463
]

# Given y values (dependent variable)
y_values = [
    0.800513, 0.679104, 0.795951, 0.675294, 0.600509, 0.316563, 0.385038, 0.827931,
    0.782503, 0.740227, 0.706465, 0.825354, 0.774482, 0.859344, 0.317341, 0.276073,
    0.622348, 0.753778, 0.81122, 0.752027, 0.760913, 0.646753, 0.644263, 0.645431,
    0.757772, 0.700966, 0.736063, 0.63276, 0.750805, 0.713226, 0.826541, 0.700504,
    0.762063, 0.62596, 0.350779, 0.748662, 0.725191, 0.688161, 0.627175, 0.631772,
    0.479683, 0.404787, 0.715098, 0.558354, 0.605257, 0.662286, 0.657183, 0.734083,
    0.794746, 0.729946, 0.532554, 0.429323, 0.833787, 0.764253, 0.674717, 0.743139,
    0.697307, 0.799114, 0.784835, 0.545508, 0.508815, 0.59194, 0.263543, 0.398551,
    0.341646, 0.269769, 0.557163, 0.689123, 0.243821, 0.263958, 0.235995, 0.218811,
    0.180673, 0.383156, 0.136236, 0.734807, 0.717774, 0.532256, 0.457291, 0.753649,
    0.635169, 0.528051, 0.626319, 0.854222, 0.281572, 0.486271, 0.636222, 0.747928,
    0.198142, 0.199919, 0.72181, 0.198472, 0.732244, 0.795921, 0.57513, 0.726263,
    0.339192, 0.435981, 0.686455, 0.223966
]

# Convert the lists to NumPy arrays
X = np.array(X_values).reshape(-1, 1)  # Reshape X into a 2D array
y = np.array(y_values)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Function to calculate confidence intervals
def get_confidence_intervals(model, X, y, num_bootstraps=1000, alpha=0.05):
    coefficients = []
    for _ in range(num_bootstraps):
        # Generate bootstrap sample indices
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        # Fit linear regression model
        model.fit(X_bootstrap, y_bootstrap)
        # Store coefficients
        coefficients.append(model.coef_[0])
    # Calculate confidence intervals
    lower_bound = np.percentile(coefficients, 100 * alpha / 2)
    upper_bound = np.percentile(coefficients, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound

# Calculate confidence intervals
lower_bound, upper_bound = get_confidence_intervals(model, X, y)

# Generate points for plotting the regression line
x_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Plot the data
plt.scatter(X, y, color='blue', label='Data')

# Plot the regression line
plt.plot(x_range, y_pred, color='red', label='Linear Regression Line')

# Plot confidence intervals
#0.88574-lower_bound
# #plt.fill_between(x_range.flatten(), 
#                  model.predict(x_range).flatten() + 0.06053100340423612, 
#                  model.predict(x_range).flatten() - 0.06053100340423612, 
#                  color='gray', alpha=0.2, label='95% Confidence Intervals')

# Add labels and legend
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Linear Regression with Confidence Intervals')
# plt.legend()

# Show plot
#plt.show()

# Create a DataFrame
df = pd.DataFrame({'X': X_values, 'y': y_values})

# Plot data points in black
plt.scatter(df['X'], df['y'], color='blue', label='Data', s = 12)

# Plot regression line in red and CI in gray
sns.regplot(x='X', y='y', data=df, ci=95, color='red', scatter=False, label='Regression Line')

# Show plot
plt.legend(labels=['Data', 'Regression Line', '95% CI'])
plt.ylabel('tan($θ_H$)')
plt.xlabel('tan($θ_S$)')
plt.title('Linear Regression with Confidence Intervals')
plt.show()