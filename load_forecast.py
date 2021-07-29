import datetime
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from tensorflow import keras

cwd = os.getcwd()

'''Loading the Data'''
def get_directories(path) -> list:
    """
    Returns a list of directories in the given path.
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
directories = get_directories('Profiles')
directories.sort()


def load_profile_df(filename) -> pd.DataFrame:
    """
    Converts xls and xlsx files to pandas dataframe.
    """
    print(filename)
    f = cwd+'/Profiles/'+filename+'/'+os.listdir(cwd+'/Profiles/'+filename)[0]
    print(f)
    if f.endswith('.xls') or f.endswith('.xlsx'):
        df = pd.read_excel(f, sheet_name=None)
        df = pd.concat(df)
        print(df)
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



#my_load_profiles, interval_column_labels = load_profile_df(directories[0])
def concat_all_excel_dfs(directories):
    full_df = pd.DataFrame()
    for d in directories:
        df, _ = load_profile_df(d)
        full_df = pd.concat([full_df, df])
    return full_df
my_load_profiles = concat_all_excel_dfs(directories)

'''Plot/Evaluate Data'''
def plot_load_data(df):
    """
    Plot the load data for each row in the dataframe."""
    df.drop(df.columns[:2], axis=1, inplace=True)
    #print(df.head())
    plt.figure(figsize=(15,5))
    plt.plot(df.iloc[1, :])
    plt.title("Loads across 24 hours")
    plt.xticks(np.arange(0, len(df.columns), 6), df.columns[::6], rotation=45)
    plt.xlabel("Interval (15 min.)")
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

@dataclass
class BaseLineModel:
    tf_dataset = tf.data.Dataset.from_tensor_slices((my_load_profiles.index.values, my_load_profiles.values))
    tf_dataset_training = tf_dataset.shuffle(buffer_size=1000).batch(1).repeat(1)
    tf_dataset_test = tf_dataset.shuffle(buffer_size=1000).batch(1).repeat(1)
    model = tfdf.keras.RandomForestModel()
    model.fit(tf_dataset_training)
    model.predict(tf_dataset_test)
    print(model.summary())



@dataclass
class Logistic_Regression:
    """Build a logistic regression model for the"""
