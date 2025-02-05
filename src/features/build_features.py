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
plt.rcParams['figuer.figsize'] = (20, 5)
plt.rcParams['figuer.dpi'] = 100

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

df_lowpass = LowPass.apply_lowpass_lfilter(df_lowpass, 'acc_y', fs, cutoff, order=5)

df_lowpass.info()

print(df_lowpass['acc_y_lowpass'].isna().sum())

subset = df_lowpass[df_lowpass['set'] == 45]

# print(subset.label)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize = (20, 10))
ax[0].plot(subset['acc_y'].reset_index(drop=True), label='row data')
ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), label='butterworth filter')
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

df_lowpass.head()

for col in predicotr_columns:
    df_lowpass = LowPass.apply_lowpass_lfilter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + '_lowpass']
    del df_lowpass[col + '_lowpass']
    


