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
def get_directories(path):
    """
    Returns a list of directories in the given path.
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def load_profile_df(filename):
    """
    Converts xls and xlsx files to pandas dataframe.
    """
    print(filename)
    f = filename+'/'+os.listdir(filename)[0]
    print(f)
    if f.endswith('.xls') or f.endswith('.xlsx'):
        df = pd.read_excel(f)
    else:
        print('File type not supported. ', f)
        sys.exit()
    return df

directories = get_directories('Profiles')
directories.sort()
#print(get_directories('Profiles'))
#for d in directories:
#    print(load_profile_df(d).head())

'''Plot'''
def plot_data(df):
    """
    Plot"""


'''Data Evaluation/Statistics'''
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

