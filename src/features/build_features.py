import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction

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

# df[df['set'] == 25]['acc_y'].plot()
# df[df['set'] == 50]['acc_y'].plot()

duration = df[df['set'] == 1].index[-1] - df[df['set'] == 1].index[0]

duration.seconds

for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]
    duration =  stop - start
    df.loc[(df['set'] == s), 'duration'] = duration.seconds
    
duration_df = df.groupby(['category'])['duration'].mean()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

LowPass = LowPassFilter()

fs = 1000 / 200
cutoff = 0.9

df_lowpass = LowPass.low_pass_filter(df_lowpass, 'acc_y', fs, cutoff, order=5)

df_lowpass.info()

subset = df_lowpass[(df_lowpass['set'] == 15)]

print(subset.label)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize = (20, 10))
ax[0].plot(subset['acc_y'].reset_index(drop=True), label='row data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), label='butterworth filter')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

df_lowpass.head()

for col in predicotr_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + '_lowpass']
    del df_lowpass[col + '_lowpass']
    
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_lowpass, predicotr_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predicotr_columns) + 1), pc_values)
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Elbow Method for Optimal PCA Components')
plt.grid()
plt.show()

df_pca = PCA.apply_pca(df_pca, predicotr_columns, 3)

subset = df_pca[df_pca['set'] == 35]

subset[['pca_1', 'pca_2', 'pca_3']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared['acc_x'] ** 2 + df_squared['acc_y'] ** 2 + df_squared['acc_z'] ** 2
gyr_r = df_squared['gyr_x'] ** 2 + df_squared['gyr_y'] ** 2 + df_squared['gyr_z'] ** 2

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

subset = df_squared[df_squared['set'] == 14]

subset[['acc_r', 'gyr_r']].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------