import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
import os
import datetime
import tensorflow as tf
from tensorflow import keras
import tensorflow_decision_forests as tfdf

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
        df.drop("ADDTIME", axis=1, inplace=True)
        df.rename(columns={"PType_WZ":"Station"}, inplace=True)
        df.drop((df.columns)[-4:], axis=1, inplace=True)
        original_columns = list(df.columns)
        print(original_columns)
        original_columns[0] = "Station"
        original_columns = ["Load at Interval {}".format(i) for i in range(1, len(original_columns[:])-1)]
        interval_column_labels = original_columns
        
        original_columns.insert(0, "Station")
        original_columns.insert(1, "Date")
        #df['Load Intervals'] = df[interval_column_labels].shift(len(interval_column_labels)-1, axis = "columns")
        #df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        print(original_columns)
        df.columns = original_columns
        print(df.columns)
        #df.groupby(df.index.day).mean()
        #df.set_index('Date', inplace=True)
        #df.set_index(interval_column_labels, inplace=True)
        
        print(df)
    else:
        print('File type not supported. ', f)
        sys.exit()
    return df, interval_column_labels

directories = get_directories('Profiles')
directories.sort()
my_load_profiles, interval_column_labels = load_profile_df(directories[0])

#for d in directories:
#    print(d)
    #print(load_profile_df(d).head())

'''Plot/Evaluate Data'''
def plot_load_data(df):
    """
    Plot the load data for each row in the dataframe."""
    df.drop(df.columns[:2], axis=1, inplace=True)
    print(df.head())
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.plot(df.iloc[1, :])
    plt.title("Loads across 24 hours")
    plt.xlabel("Hour")
    plt.ylabel("Load (kWh)")
    plt.show()
    df.plot()
    


print("MY LOAD DATAFRAME: ")
print(my_load_profiles.head())
print(my_load_profiles.shape)
print(my_load_profiles.columns)
print(my_load_profiles.index)
plot_load_data(my_load_profiles)



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
print('MOVING AVERAGE')
#print(moving_average(my_load_profiles, 15).head())

def get_seasonal_avg(df, season):
    """
    Calculates the seasonal average of a given dataframe.
    """
    return df.groupby(df.index.month).mean()
print("SEASONAL AVERAGE")
#print(get_seasonal_avg(my_load_profiles, 1))

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
tf_dataset = tf.data.Dataset.from_tensor_slices((my_load_profiles.index.values, my_load_profiles.values))
model = tfdf.keras.RandomForestModel()
model.fit(tf_dataset)
print(model.summary())


@dataclass
class Forecast_Model(training_data, validation_data):
    """
    Time-Series Forecast Model.
    """
    forecast_model: object
    forecast_model_name: str
    forecast_model_data: pd.DataFrame
    forecast_model_data_columns: list

    class baseline_model(Forecast_Model):
        def __init__(self):
            pass

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

