import os
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
import datetime as dt
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def percent_purchase(df, unit_of_time):
    """
    Filter df to get count of visits & purchases by given unit of time.
    
    Input:
    df -- The dataframe to filter
    unit_of_time -- string indicating what unit of time to group observations by.  Monthly, weekly, etc.
    
    Output:
    Dataframe resampled by unit of time.
    """
    df['date'] = df['date'].astype(str)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    #Get number of purchases per time unit
    df_ts = df[['date', 'purchase']]
    df_ts_gr = df_ts.groupby('date').sum()
    purchases = df_ts_gr.resample(unit_of_time).sum()
    
    #Get number of visits per time unit
    df_ts = df[['date', 'purchase']]
    df_ts_gr = df_ts.groupby('date').count()
    visits = df_ts_gr.resample(unit_of_time).sum()
    visits['visits'] = visits['purchase']
    visits.drop('purchase', axis=1, inplace=True)
    
    #Combine purchases and visits
    df_time = purchases.join(visits, how='inner')
    
    #Calculate percent
    df_time['percent_purchase'] = df_time['purchase'] / df_time['visits'] * 100
    
    return df_time

def graph_roll_avg_percent_purchase(df, window):
    """
    Graph rolling average of the percent of visits that lead to a purchase
    
    Input:
    df -- Dataframe in time series format
    window -- Time period over which to compute rolling average
    
    Output:
    Displays line graph of rolling average.  No return value
    """
    rolling_mean = df.rolling(window=window, center=False).mean()
    plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(x=rolling_mean.index, y="percent_purchase", data=rolling_mean)
    plt.ylabel('% Purchase')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.title('Percent Purchase - 4 Week Rolling Average')
    plt.rcParams.update({'font.size': 22})


def graph_roll_avg_visits(df, window):
    """
    Graph rolling average of number of visits to the store
    
    Input:
    df -- Dataframe in time series format
    window -- Time period over which to compute rolling average
    
    Output:
    Displays line graph of rolling average.  No return value
    """
    rolling_mean = df.rolling(window=window, center=False).mean()
    plt.subplots(figsize=(12, 6))
    ax = sns.lineplot(x=rolling_mean.index, y="visits", data=rolling_mean)
    plt.ylabel('# of Visits')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.title('Weekly Visits- 4 Week Rolling Average')
    plt.rcParams.update({'font.size': 22})