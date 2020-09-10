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
from joblib import dump, load
from tqdm import tqdm
import warnings

def filter_us(csv_path):
    """
    Filter raw data to only include visits from the US and save as .csv file
    
    Input:
    csv_path -- string of file path to original data set.
    
    Output:
    No return value.  Saves file as data1.csv.
    """
    df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'})
    df_us = df.loc[df['geoNetwork'].astype(str).str.contains('United States')]
    df_us.to_csv('../Data/data1.csv')
    del df
    del df_us
    
def unpack_df(csv_path='../Data/data1.csv', nrows=None):
    """
    Unpack the nested structure of JSON columns
    
    *Source*
    Much of the code here was created by Kaggler Julian Peller
    https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
    ***
    
    Input:
    csv_path -- Path to csv file to unpack
    nrows -- Number of rows to unpack
    
    Output:
    Returns a copy of the csv as a dataframe with each nested element as a new column.
    """
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

def process_data(df):
    """
    Complete all data cleaning steps.
    
    Input:
    df -- DataFrame containing unpacked data
    
    Output:
    No return.  Saves processed dataframe as data2.csv.
    """
    #Drop unneeded columns
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop(['customDimensions', 'device.browserSize', 'device.browserVersion', 'device.flashVersion'], axis=1, inplace=True)
    for column in df.columns:
        if df[column].value_counts().index[0] == 'not available in demo dataset':
            df.drop(column, axis=1, inplace=True)
    
    #Update Data Types
    
    #Date to datetime
    df['date'] = df['date'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    #Bin start times
    df['time'] = df['visitStartTime'].apply(lambda x: dt.datetime.utcfromtimestamp(float(x)))
    bin_values = [0,4,8,12,16,20,24]
    bin_labels = ['Late Night', 'Early Morning','Morning','Afternoon','Evening','Night']
    df['hour'] = df['time'].dt.hour
    df['time_of_day'] = pd.cut(df['hour'], bins=bin_values, labels=bin_labels, include_lowest=True)
    df.drop(['hour', 'visitStartTime'], axis=1, inplace=True)
    
    #Group unique categorical values
    value_counts_list = []
    for column in df.columns:
        value_counts_list.append(df[column].value_counts())
        
    #Group Browsers
    top_browsers = list(value_counts_list[7].index[0:10])
    df['device.browser'] = df['device.browser'].apply(lambda x: x if x in top_browsers else 'Other')
    
    #Group OS
    top_os = list(value_counts_list[8].index[0:6])
    df['device.operatingSystem'] = df['device.operatingSystem'].apply(lambda x: x if x in top_os else 'Other')
    
    #Group Ad Content
    top_ad_content = list(value_counts_list[31].index[0:9])
    df['trafficSource.adContent'] = df['trafficSource.adContent'].apply(lambda x: x if x in top_ad_content else 'Other')

    #Group Ad Campaign
    top_camp = list(value_counts_list[26].index[0:8])
    df['trafficSource.campaign'] = df['trafficSource.campaign'].apply(lambda x: x if x in top_camp else 'Other')
    df['trafficSource.campaign'].replace('(not set)', 'None', inplace=True)
    
    #Group Keywords
    top_keyword = list(value_counts_list[30].index[0:11])
    df['trafficSource.keyword'] = df['trafficSource.keyword'].apply(lambda x: x if x in top_keyword else 'Other')
    df['trafficSource.keyword'].replace('(not provided)', 'None', inplace=True)
    
    #Consolidate medium values
    df['trafficSource.medium'].replace('(none)', 'None', inplace=True)
    df['trafficSource.medium'].replace('(not set)', 'None', inplace=True)
    
    #Group referral path
    top_path = list(value_counts_list[25].index[0:22])
    df['trafficSource.referralPath'] = df['trafficSource.referralPath'].apply(lambda x: x if x in top_path else 'Other')
    df['trafficSource.referralPath'].replace('/', 'None', inplace=True)
    
    #Group Source
    top_source = list(value_counts_list[27].index[0:22])
    df['trafficSource.source'] = df['trafficSource.source'].apply(lambda x: x if x in top_source else 'Other')
    
    #Drop columns with no information
    df.drop('socialEngagementType', axis=1, inplace=True)
    df.drop('totals.bounces', axis=1, inplace=True)
    df.drop('totals.newVisits', axis=1, inplace=True)
    df.drop('totals.visits', axis=1, inplace=True)
    df.drop('trafficSource.adwordsClickInfo.gclId', axis=1, inplace=True)
    df.drop('trafficSource.adwordsClickInfo.isVideoAd', axis=1, inplace=True)
    df.drop('trafficSource.isTrueDirect', axis=1, inplace=True)
    
    #Resolve NaN Values
    
    #Replace missing values with -1
    df['totals.sessionQualityDim'].fillna(-1, inplace=True)
    
    # Fill in 0s for transaction/revenue data
    df['totals.totalTransactionRevenue'].fillna(0.0, inplace=True)
    df['totals.transactionRevenue'].fillna(0.0, inplace=True)
    df['totals.transactions'].fillna(0.0, inplace=True)
    
    # Fill time on site NaN's using mean.  Different for purchasers vs not.
    no_purchase_tos = df.loc[df['totals.totalTransactionRevenue'] == 0.0]['totals.timeOnSite']
    no_purchase_tos.dropna(inplace=True)
    no_purchase_time = no_purchase_tos.astype(int).mean()

    purchase_tos = df.loc[df['totals.totalTransactionRevenue'] != 0.0]['totals.timeOnSite']
    purchase_tos.dropna(inplace=True)
    purchase_time = purchase_tos.astype(int).mean()
    
    df.loc[df['totals.transactionRevenue'].astype(int) > 0, 'purchase'] = True
    df.loc[df['totals.transactionRevenue'].astype(int) == 0, 'purchase'] = False
    df['purchase'] = df['purchase'].astype(bool)
    df['totals.timeOnSite'].fillna(df['purchase'].apply(lambda x: purchase_time if x==True else no_purchase_time), inplace=True)
    
    #Fill page views
    df['totals.pageviews'].fillna(1.0, inplace=True)
    
    #Fill AdWords
    df['trafficSource.adwordsClickInfo.adNetworkType'].fillna('None', inplace=True)
    df['trafficSource.adwordsClickInfo.slot'].fillna('None', inplace=True)
    df['trafficSource.adwordsClickInfo.page'].fillna(0.0, inplace=True)
    
    #Totals to integers
    for column in df.columns:
        if 'totals.' in str(column):
            df[column] = df[column].astype(int)
    
    #Revenue to USD
    df['revenue_usd'] = df['totals.transactionRevenue'] / (10**6)
    df['total_revenue_usd'] = df['totals.totalTransactionRevenue'] / (10**6)
    
    #Save to csv
    df.to_csv('../Data/data2.csv')
    
    
######################
# The following functions are created for use in the load_hits_df function.

def todict(dic, key, value):
    """
    Nested structure to dictionary
    
    Input:
    dic -- dictionary-like object (JSON)
    key -- dict key
    value -- dict value
    
    Output:
    Dictionary object
    """
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]
    return dic


def resolve_json(hitsdic, hits_json, key='NoneName'):
    """
    Resolve nested structure of JSON data
    
    Returns dictionary object.
    """
    if type(hits_json) == list:
        if len(hits_json) == 0:
            pass
        else:
            for subjson in hits_json:
                hitsdic = resolve_json(hitsdic, subjson)
    elif type(hits_json) == dict:
        for i in hits_json.keys():
            hitsdic = resolve_json(hitsdic, hits_json[i],i)
    else:
        hitsdic = todict(hitsdic, key, hits_json)
    return hitsdic


def complex_replace(x):
    """
    Replace/standardize special characters
    
    Input:
    x -- String
    
    Output:
    Returns string with standardized special characters.
    """
    dic = {}
    return resolve_json(dic, json.loads(x.replace('\'','\"'). \
                                        replace('TRUE','true'). \
                                        replace('True','true'). \
                                        replace('FALSE','false'). \
                                        replace('False','false'). \
                                        replace(', \"',', !&~'). \
                                        replace('\", ','!&~, '). \
                                        replace('\": ','!&~: '). \
                                        replace(': \"',': !&~'). \
                                        replace(' {\"',' {!&~'). \
                                        replace('\"}, ','!&~}, '). \
                                        replace('[{\"','[{!&~'). \
                                        replace('\"}]','!&~}]'). \
                                        replace('\"','_'). \
                                        replace('!&~','\"'). \
                                        encode('gbk','ignore'). \
                                        decode('utf-8','ignore'). \
                                        replace('\\','')))


def replace(x):
    """
    JSON to Dictionary
    
    Input:
    x -- JSON formatted object
    
    Output:
    Returns Dictionary object
    """
    return  json.loads(x)


def load_hits_df(csv_path, nrows=None, chunksize=10_000, percent=100):
    """
    Unpack nested structure of the 'hits' column.
    
    Input:
    csv_path -- string of file to unpack
    nrows -- int or None.  Number of rows to unpack.
    chunksize -- int.  How many rows to process at a time.  Lower for less memory usage.
    percent -- int.  Percent of rows to unpack.
    
    Output:
    Returns pandas dataframe with hits variables as separate columns.  DF also saved as data4.csv
    """
    n=1
    df_list = []
    feature = ['hits']
    chunk = pd.read_csv(csv_path,
                        nrows=nrows, 
                        chunksize=chunksize, 
                        dtype={'fullVisitorId': 'str'}) # Important!!
    for subchunk in chunk:
        for column in feature:
            if column in ['hits']:
                column_as_df = json_normalize(subchunk[column].apply(complex_replace))
            else:
                column_as_df = json_normalize(subchunk[column].apply(replace))
            column_as_df.columns = [f'{column}_{subcolumn}' for subcolumn in column_as_df.columns]
            subchunk.drop(column, axis=1, inplace=True)
            subchunk = subchunk.reset_index(drop=True).merge(column_as_df,
                                           right_index=True,
                                           left_index=True)
        n = n+1
        df_list.append(subchunk.astype('str'))
        del column_as_df, subchunk
    return pd.concat(df_list, ignore_index=True, sort=True)
######################
def add_econ_data(csv_path):
    """
    Add CCI, USDI, and S&P 500 data to the DataFrame
    
    Input:
    csv_path -- string.  File path to data source
    
    Output:
    returns dataframe including economic data.
    """
    #Read in data
    df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'})
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    #Format date
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].dt.date
    df['Month_Year'] = df['date'].apply(lambda x: str(x.month) + '-' + str(x.year))
    
    #Add cci data
    cci = pd.read_csv('../Data/Econ_Data/CCI.csv')
    cci = cci[['TIME', 'Value']]
    cci['CCI'] = cci['Value']
    cci['TIME'] = pd.to_datetime(cci['TIME'])
    cci['Month'] = cci['TIME'].dt.month
    cci['Year'] = cci['TIME'].dt.year
    cci['Month_Year'] = cci['Month'].astype(str) + '-' + cci['Year'].astype(str)
    cci.drop(['Value', 'TIME', 'Month', 'Year'], axis=1, inplace=True)
    
    #Add S&P data
    snp = pd.read_csv('../Data/Econ_Data/snp500.csv')
    snp['S&P'] = snp['Close']
    snp = snp[['Date', 'S&P']]
    snp['Date'] = pd.to_datetime(snp['Date'])
    snp.set_index('Date', inplace=True)
    snp = snp.asfreq('D', method='ffill')
    snp.sort_index(inplace=True)
    
    
    #Add USDI data
    usdi = pd.read_csv('../Data/Econ_Data/US Dollar Index Futures Historical Data.csv')
    usdi['USDI'] = usdi['Price']
    usdi = usdi[['Date', 'USDI']]
    usdi['Date'] = pd.to_datetime(usdi['Date'])
    usdi.set_index('Date', inplace=True)
    usdi = usdi.resample('D').ffill()
    usdi.sort_index(inplace=True)
    
    #Merge DataFrames
    #Merge CCI by month
    df = df.merge(cci, how='left', on='Month_Year')
    df.drop('Month_Year', axis=1, inplace=True)
    #Merge S&P by Date
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df = df.merge(snp, how='left', left_on='date', right_on='Date')
#     df.drop('Date', axis=1, inplace=True)
    #Merge USDI
    df = df.merge(usdi, how='left', left_on='date', right_on='Date')
#     df.drop('Date', axis=1, inplace=True)
    
    return df
    


def processing_for_modeling(csv_path):
    """
    Clean data3.csv for modeling
    
    Input:
    csv_path -- Path to data3.csv file
    
    Output:
    returns filtered df ready for modeling
    """
    
    df2 = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'})
    
    #Drop columns with no information
    df2.drop('Unnamed: 0', axis=1, inplace=True)
    df2.drop(['geoNetwork.continent', 'geoNetwork.subContinent', 'geoNetwork.country'], axis=1, inplace=True)

    #Group network domain column
    top_net_domain = list(df2['geoNetwork.networkDomain'].value_counts().head(31).index)
    df2['geoNetwork.networkDomain'] = df2['geoNetwork.networkDomain'].apply(lambda x: x if x in top_net_domain else 'Other')
    df2['geoNetwork.networkDomain'].replace('(not set)', 'None', inplace=True)
    
    #Drop columns not needed in modeling stage
    df2.drop(['time', 'totals.totalTransactionRevenue', 'totals.transactionRevenue', 'date', 'fullVisitorId', 'visitId', 'total_revenue_usd', 'totals.transactions', 'purchase'], axis=1, inplace=True)
    
    #Get product subcategory of last viewed item
    df2['hits_v2ProductCategory'].fillna("'None'", inplace=True)
    df2['product_category'] = df2['hits_v2ProductCategory'].apply(lambda x: str(x).split("'")[1])
    df2['product_category'].replace("(not set)", "None", inplace=True)
    df2['product_category'] = df2['product_category'].apply(lambda x: str(x).split("/")[1] if "/" in str(x) else str(x))
    top_prod_cats = list(df2['product_category'].value_counts().head(14).index)
    df2['product_category'] = df2['product_category'].apply(lambda x: x if x in top_prod_cats else 'Other')

    #Get price of last viewed item
    df2['hits_productPrice'].fillna("'0'", inplace=True)
    df2['product_price'] = df2['hits_productPrice'].apply(lambda x: str(x).split("'")[1])
    df2['product_price'] = df2['product_price'].astype(int)
    df2['product_price'] = df2['product_price'] / (10**6)
    
    #Drop unneeded hits columns
    unused_hits = []
    for col in df2.columns:
        if 'hits_' in col:
            unused_hits.append(col)
    df2.drop(unused_hits, axis=1, inplace=True)
    df2.drop('Unnamed: 0.1', axis=1, inplace=True)
    
    df2.to_csv('../Data/data4.csv')
    
    return df2