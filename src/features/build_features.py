import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle('../../data/interim/02_outliers_removed_chauvenet.pkl')

predicotr_columns = list(df.columns[:6])

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5) 
plt.rcParams['figure.dpi'] = 100        
 
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predicotr_columns:
    df[col] = df[col].interpolate()
    
df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df['set'] == 25]['acc_y'].plot()
df[df['set'] == 50]['acc_y'].plot()

# Calculate duration for the first set
duration = df[df['set'] == 1].index[-1] - df[df['set'] == 1].index[0]
duration.seconds  

# Calculate duration for each set
for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]
    
    # Calculate duration and add to dataframe
    duration = stop - start
    df.loc[(df['set'] == s), 'duration'] = duration.seconds

# Calculate mean duration for each category
duration_df = df.groupby(['category'])['duration'].mean()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
    
# Create a copy of the dataframe for low-pass filtering
df_lowpass = df.copy()

# Initialize Low Pass Filter
LowPass = LowPassFilter()

# Calculate sampling frequency
fs = 1000 / 200
cutoff = 0.9

# Apply low-pass filter to acc_y column
df_lowpass = LowPass.low_pass_filter(df_lowpass, 'acc_y', fs, cutoff, order=5)

# Display dataframe information
df_lowpass.info()

# Extract subset for set 15
subset = df_lowpass[(df_lowpass['set'] == 15)]

# Visualize original and filtered data
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset['acc_y'].reset_index(drop=True), label='raw data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), label='butterworth filter')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# View first few rows
df_lowpass.head()

# Apply low-pass filter to all predictor columns
for col in predicotr_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + '_lowpass']
    del df_lowpass[col + '_lowpass']
    
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

# Create a copy of the dataframe for PCA
df_pca = df_lowpass.copy()

# Initialize Principal Component Analysis
PCA = PrincipalComponentAnalysis()

# Determine explained variance for principal components
pc_values = PCA.determine_pc_explained_variance(df_lowpass, predicotr_columns)

# Plot explained variance to find optimal number of components
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predicotr_columns) + 1), pc_values, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Elbow Method for Optimal PCA Components')
plt.grid()
plt.show()

# Apply PCA with 3 components
df_pca = PCA.apply_pca(df_pca, predicotr_columns, 3)

# Extract subset for set 35
subset = df_pca[df_pca['set'] == 35]

subset[['pca_1', 'pca_2', 'pca_3']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# Create a copy of the PCA dataframe
df_squared = df_pca.copy()

# Calculate resultant acceleration (magnitude) using Pythagorean theorem
acc_r = df_squared['acc_x'] ** 2 + df_squared['acc_y'] ** 2 + df_squared['acc_z'] ** 2

# Calculate resultant gyroscope (magnitude) using Pythagorean theorem
gyr_r = df_squared['gyr_x'] ** 2 + df_squared['gyr_y'] ** 2 + df_squared['gyr_z'] ** 2

# Take square root to get final magnitudes
df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

# Extract subset for set 14
subset = df_squared[df_squared['set'] == 14]

subset[['acc_r', 'gyr_r']].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

# Create a copy of the squared dataframe
df_temporal = df_squared.copy()

# Initialize Numerical Abstraction
numAbs = NumericalAbstraction()

# Add resultant vectors to predictor columns
predicotr_columns = predicotr_columns + ['acc_r', 'gyr_r']

# Set window size for temporal abstraction
ws = 5

# Initialize list to store processed sets
df_temporal_list = []

# Process each set separately to avoid boundary effects
for s in df_temporal['set'].unique():
    print(f'Applying Numerical Abstraction for set {s}')
    
    # Extract subset for current set
    subset = df_temporal[df_temporal['set'] == s].copy()
    
    # Calculate mean over sliding window
    df_temporal = numAbs.abstract_numerical(df_temporal, predicotr_columns, ws, 'mean')
    
    # Calculate standard deviation over sliding window
    df_temporal = numAbs.abstract_numerical(df_temporal, predicotr_columns, ws, 'std')
    
    # Add processed subset to list
    df_temporal_list.append(subset)

# Combine all processed sets
df_temporal = pd.concat(df_temporal_list)

# Plot original, mean, and std for accelerometer y-axis
subset[['acc_y', 'acc_y_temp_mean_ws_5', 'acc_y_temp_std_ws_5']].plot()

# Plot original, mean, and std for gyroscope y-axis
subset[['gyr_y', 'gyr_y_temp_mean_ws_5', 'gyr_y_temp_std_ws_5']].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

# Create a copy of temporal dataframe and reset index
df_freq = df_temporal.copy().reset_index()

# Initialize Fourier Transformation
freqAbs = FourierTransformation()

# Calculate sampling frequency and window size
sf = int(1000 / 200)  # Sampling frequency in Hz
ws = int(2800 / 200)  # Window size in samples

# Test Fourier transformation on acc_y
freqAbs.abstract_frequency(df_freq, ['acc_y'], ws, fs)

# Extract subset for visualization
subset = df_freq[df_freq['set'] == 15]

# Plot original accelerometer data
subset[['acc_y']].plot()

# Plot all frequency features for acc_y
subset[
    [
        'acc_y',                    # Original signal
        'acc_y_max_freq',           # Maximum frequency
        'acc_y_freq_weighted',      # Weighted frequency
        'acc_y_pse',               # Power spectral entropy
        'acc_y_freq_1.429_Hz_ws_14',  # Power at 1.0 Hz
        'acc_y_freq_2.5_Hz_ws_14'   # Power at 2.5 Hz
    ]
].plot()

# Initialize list to store processed sets
df_freq_list = []

# Process each set separately to avoid boundary effects
for s in df_freq['set'].unique():
    print(f'Applying Fourier Transformation for set {s}')
    
    # Extract and reset index for current set
    subset = df_freq[df_freq['set'] == s].reset_index(drop=True).copy()
    
    # Apply Fourier transformation to all predictor columns
    subset = freqAbs.abstract_frequency(subset, predicotr_columns, ws, sf)
    
    # Add processed subset to list
    df_freq_list.append(subset)

# Combine all processed sets and set the index
df_freq = pd.concat(df_freq_list).set_index('epoch (ms)', drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# Drop any rows containing NaN values
df_freq = df_freq.dropna()

# Downsample the data by taking every second row
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

# Create a copy of the frequency dataframe to preserve original data
df_cluster = df_freq.copy()

# Define columns to use for clustering
cluster_columns = ['acc_x', 'acc_y', 'acc_z']

# Define range of K values to test for optimal clustering
k_values = range(2, 10)
inertias = []


# Perform K-means clustering for different K values to find optimal clusters
for k in k_values:
    # Select subset of data for clustering
    subset = df_cluster[cluster_columns]
    
    # Initialize K-means with current K value
    k_means = KMeans(n_clusters=k, n_init=20, random_state=0)
    
    # Fit model and predict cluster labels
    cluster_labels = k_means.fit_predict(subset)
    
    # Store inertia (sum of squared distances) for each K
    inertias.append(k_means.inertia_)


# Plot elbow curve to visualize optimal number of clusters
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('K')
plt.ylabel('Sum of squared distances')
plt.grid()
plt.show()


# Perform final clustering with chosen K (5 clusters)
k_means = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster['cluster'] = k_means.fit_predict(subset)

# 3D scatter plot of clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

# Plot each cluster with different color
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=c)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.legend()
plt.savefig('../../reports/figures/cluster_plot.png')
plt.show()


# 3D scatter plot comparing clusters by original label
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')

# Plot each original label with different color
for l in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == l]
    ax.scatter(subset['acc_x'], subset['acc_y'], subset['acc_z'], label=l)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.legend()
plt.savefig('../../reports/figures/cluster_label_plot.png')
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle('../../data/interim/03_data_features.pkl')