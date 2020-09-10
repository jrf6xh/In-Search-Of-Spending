import os
import sys
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
import datetime as dt
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, make_scorer, silhouette_score, silhouette_samples, calinski_harabasz_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from joblib import dump, load
from tqdm import tqdm
import warnings

def add_clustering(data, reduced_data, clusters):
    #Create best model
    best_cluster = KMeans(n_clusters=clusters, random_state=70, algorithm='full')

    #Fit model.
    best_cluster_fit = best_cluster.fit(reduced_data)
    print('Done Fitting Model')
    
    #Get labels for use in silhouette score calculations
    best_cluster_labels = best_cluster_fit.predict(reduced_data)
    print('Done Predicting')
    
    sil_scores = silhouette_samples(reduced_data, best_cluster_labels)
    print('Done with Silhouette Scores')

    #Save the silhouette scores
    dump(sil_scores, '../Models/' + 'sil_scores_kmeans.joblib') 

    #Save the model
    dump(best_cluster_fit, '../Models/' + 'kmeans_100_fit.joblib')

    #Save the predictions
    dump(best_cluster_labels, '../Models/' + 'kmeans_100_labels.joblib')
    
    #Turn into series objects
    sil_scores = pd.Series(sil_scores, name='sil_score')
    best_cluster_labels = pd.Series(best_cluster_labels, name='cluster_label')
    
    #Merge with original data
    data_merged = data.reset_index()
    data_merged.drop('index', axis=1, inplace=True)
    to_concat = [data_merged, sil_scores, best_cluster_labels]
    data_merged = pd.concat(to_concat, axis=1)
    
    return data_merged
    
    

