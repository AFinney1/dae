import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
import os
import datetime
import tensorflow as tf
from tensorflow import keras


cwd = os.getcwd()

'''Loading the Data'''
def get_directories(path) -> list:
    """
    Returns a list of directories in the given path.
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def load_profile_df(filename) -> pd.DataFrame:
    """
    Converts xls and xlsx files to pandas dataframe.
    """
    print(filename)
    f = cwd+'/Profiles/'+filename+'/'+os.listdir(cwd+'/Profiles/'+filename)[0]
    print(f)
    if f.endswith('.xls') or f.endswith('.xlsx'):
        df = pd.read_excel(f)
    else:
        print('File type not supported. ', f)
        sys.exit()
    return df

directories = get_directories('Profiles')
directories.sort()

for d in directories:
    print(d)
    #print(load_profile_df(d).head())

'''Plot/Evaluate Data'''
def plot_load_data(df):
    """
    Plot the load data for each day over time.
    """
    
    original_columns = list(df.columns)
    df.drop("ADDTIME", axis=1, inplace=True)
    print(original_columns)
    #original_columns[0] = "Station"
    original_columns = ["Load at 15 min {}".format(i) for i in range(1, len(original_columns[2:]))]
    original_columns.insert(0, "Station")
    original_columns.insert(1, "Date")
    #df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df.columns = original_columns
    #df.set_index('Date', inplace=True)
    #df.T
    #date_column = df['Date']
    
    df.plot(
        y = original_columns[2:],
        subplots = False
    )#columns = date_column)
    plt.title('Load Data')
    plt.xlabel('Time (15 min. Intervals)')
    plt.ylabel('Load (kWh)')
    plt.show()

    
plot_load_data(load_profile_df(directories[0]))

'''Prepare Data'''

'''Preliminary Statistics'''
def check_stationarity():
    """
    Check for constant mean and variance, and covariance is independent of time.
    """
    pass

def moving_average(df, window):
    """
    Calculates the moving average of a given dataframe.
    """
    return df.rolling(window=window).mean()

def get_seasonal_avg(df, season):
    """
    Calculates the seasonal average of a given dataframe.
    """
    return df.groupby(df.index.month).mean()

def plot_seasonal_avg(df, season):
    """
    Plots the seasonal average of a given dataframe.
    """
    plt.figure()
    plt.plot(df.groupby(df.index.month).mean())
    plt.title('Seasonal Average')
    plt.xlabel('Month')
    plt.ylabel('Temperature (C)')
    plt.show()

'''Model Building'''

@dataclass
class Forecast_Model:
    """
    Time-Series Forecast Model.
    """
    forecast_model: object
    forecast_model_name: str
    forecast_model_data: pd.DataFrame
    forecast_model_data_columns: list

    class baseline_model(Forecast_Model):
        super()

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

