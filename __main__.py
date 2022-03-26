import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

from RNN import RNN
#from CNN import CNN
from Transformer import Transformer

from utils import series_to_supervised


## Testing on open data
## Link to dataset : https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption


## Pre-processing steps to get cleaned data
## https://machinelearningmastery.com/multi-step-time-series-forecasting-with-machine-learning-models-for-household-electricity-consumption/


dataset = pd.read_csv('data/household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

# resample data to daily
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()

# We choose to keep only Global_active_power
to_drop = ['Global_reactive_power', 'Voltage',
       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
       'Sub_metering_3', 'sub_metering_4']
daily_data.drop(columns=to_drop, inplace=True)

# add calendar-related features
daily_data['day'] = pd.DatetimeIndex(daily_data.index).day
daily_data['weekday'] = ((pd.DatetimeIndex(daily_data.index).dayofweek) // 5 == 1).astype(float)
daily_data['season'] = [month%12 // 3 + 1 for month in pd.DatetimeIndex(daily_data.index).month]

# summarize
print(daily_data.info())
#print(daily_data.head())

look_back = 7
n_features = daily_data.shape[1]

# Walk-forward data split to avoid data leakage
X_train, y_train, X_test, y_test, scale_X = series_to_supervised(daily_data, train_size=0.8, n_in=look_back, n_out=7, target_column='Global_active_power', dropnan=True, scale_X=True)

# reshape input to be 3D [samples, timesteps, features]
X_train_reshaped = X_train.values.reshape((-1,look_back,n_features))
X_test_reshaped = X_test.values.reshape((-1,look_back,n_features))

y_train_reshaped = y_train.values
y_test_reshaped = y_test.values

## Testing the RNN-LSTM
"""
rnn = RNN()
rnn.train(X_train_reshaped,y_train_reshaped)
_, rmse_result, mae_result, smape_result, r2_result = rnn.evaluate(X_test_reshaped,y_test_reshaped)"""

## Testing the Transformer
tr = Transformer()
tr.train(X_train_reshaped,y_train_reshaped)
_, rmse_result, mae_result, smape_result, r2_result = tr.evaluate(X_test_reshaped,y_test_reshaped)


print('Result \n RMSE = %.2f [kWh] \n MAE = %.2f [kWh]\n R2 = %.1f [%%]' % (rmse_result,
                                                                            mae_result,
                                                                            r2_result*100))
