# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 22:45:43 2016

@author: ppradeep
"""

# -----------------------------------------------------------------------------------------
## Unsupervised RA
# -----------------------------------------------------------------------------------------

import os
clear = lambda: os.system('cls')
clear()

import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np 
from sklearn.feature_selection import VarianceThreshold 

import sklearn.metrics as sm 
from sklearn.metrics import r2_score, accuracy_score
from sklearn import preprocessing

from sklearn.cluster import KMeans 
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
#%%
## User-defined functions
def selectFeatures_VarThresh(X, threshold):
    sel = VarianceThreshold(threshold=(threshold*(1-threshold))) 
    X_sel = sel.fit_transform(X)
    # Convert it into a dataframe 
    x_tr = pd.DataFrame(X_sel, index = X.index) 
    x_tr.columns = X.columns[sel.get_support(indices = True)]
    return x_tr

## Remove culumns with >80% correlation
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return dataset

# Normalize descriptors: Transform variables to mean=0, variance=1
def normalizeDescriptors(X):
    x_norm = preprocessing.scale(X)
    x_norm = pd.DataFrame(x_norm, index = X.index) 
    x_norm.columns = X.columns
    return x_norm

#%%
path = 'C:/Users/ppradeep/Desktop/HTTK/' 
path = 'Z:/Projects/HTTK/' 

#%%
###########################################################################
## Read and analyze input data 
###########################################################################
data1 = pd.read_csv(path+'data/Prachi-112117.txt', index_col = 'CAS').loc[:,['All.Compound.Names', 'Human.Funbound.plasma', 'Human.Clint']]
data1.rename(columns={'All.Compound.Names' : 'Name'}, inplace = True)
data2 = pd.read_excel(path+'data/AFFINITY_Model_Results-2018-02-27.xlsx', index_col = 'CAS').loc[:,['Name','Fup.Med']]
data2.rename(columns={'Name': 'All.Compound.Names','Fup.Med':'Human.Funbound.plasma'}, inplace = True)
data3 = pd.read_excel(path+'data/CLint-2018-03-01-Results.xlsx', index_col = 'CAS').loc[:,['Name','CLint.1uM.Median']]
data3.rename(columns={'Name': 'All.Compound.Names','CLint.1uM.Median':'Human.Clint'}, inplace = True)

# Set y variable
y_var = 'Human.Clint'
# Create a new dataframe with chemical names and y variable value based on raw data
casList = list(set(data1.index.tolist()+data2.index.tolist()+data3.index.tolist()))
data = pd.DataFrame(index = casList, columns = ['Name',y_var])
# Update the training data. If y value is available from later data (data 2 or 3) use that, if not use from old data (data1)
for cas in data.index:
    try:
        if cas in data1.index:
            data.loc[cas,'Name'] = data1.loc[cas,'Name']
            data.loc[cas,y_var] = data1.loc[cas,y_var]
        if cas in data2.index:
            data.loc[cas,'Name'] = data2.loc[cas,'Name']
            data.loc[cas,y_var] = data2.loc[cas,y_var]
    except:
        pass
data.dropna(inplace = True) #Retain data with y variable values

## Extract y data
Y = data[y_var]

###########################################################################
## Read fingerprints
###########################################################################
fp = pd.read_csv(path+'data/CombinedFP.csv', index_col='row ID') 

###########################################################################
## Combined FP distances 
###########################################################################
dist = pd.read_csv(path+'data/CombinedFPDistances.csv', index_col='row ID') 
dist.columns = dist.index 

#%%
###########################################################################
## Cluster-based regression (medium clearance)
###########################################################################

## Extract y data for regression
Y_reg = Y[(Y > 0.9) & (Y <= 50)]
## Transform Y
Y_reg = Y_reg.apply(lambda x: np.log10(x))

## Combine all the descriptors
data = pd.concat([Y_reg, fp], axis=1).dropna(axis=0, how='any')
X_model = data.ix[:, data.columns != y_var]
Y_model = data.ix[:, data.columns == y_var]

#%%
## Build clusters and read-across
k = 20
target_predicted_count = pd.DataFrame(index = Y_model.index, columns = ['Clint1', 'Clint2','Clint3','Clint4','Clint5'])
target_predicted_threshold = pd.DataFrame(index = Y_model.index, columns = ['Clint'])

for target_idx in X_model.index:
    target = X_model.loc[target_idx,:]
    X_analog_pool = X_model[X_model.index != target_idx]
    
    # K-means clustering on the analog_pool data
    kmeans = KMeans(init='k-means++', n_clusters = k) 
    kmeans.fit(X_analog_pool) 
    k_means_labels = kmeans.labels_ 
    k_means_cluster_centers = kmeans.cluster_centers_ 
    k_means_labels_unique = np.unique(k_means_labels) 
    x_train_kmeans = kmeans.predict(X_analog_pool) 
    
    ## Create a new dataframe where each chemical is represented by a cluster number  
    ## and its associated Clearance value 
    df_kmeans = pd.DataFrame(x_train_kmeans,  index = X_analog_pool.index, columns = {'Cluster'}) 
    df_kmeans[y_var] = Y_model
    
    ## Assign target to a cluster
    target_cluster_no = int(kmeans.predict(target))
    analog_pool = df_kmeans[df_kmeans['Cluster'] == target_cluster_no].index.tolist()
    distances = dist.loc[target_idx, X_analog_pool.index].sort_values(ascending=False)
    
    # All analogs less than threshold
    analog_idx = distances[(distances > 0.70)].index.tolist()
    if len(analog_idx) == 0:
        pass
    else:
        pred = np.average([Y_model.loc[analog] for analog in analog_idx])
        target_predicted_threshold.loc[target_idx, 'Clint'] = pred        

    # All analogs less than threshold and count
    for n in range(1,6):
        analog_idx = distances[(distances > 0.70)].index.tolist()[0:n]
        if len(analog_idx) == 0:
            pass
        else:
            pred = np.average([Y_model.loc[analog] for analog in analog_idx])
            target_predicted_count.loc[target_idx, 'Clint%d' %n] = pred  
        
#%%
target_predicted_count = target_predicted_count.dropna()
target_predicted_threshold = target_predicted_threshold.dropna()

#%%
read_across_metrics1 = pd.DataFrame(columns = {'Value'})
read_across_metrics1.loc['MAE', 'Value'] = round(sm.mean_absolute_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold),2)
read_across_metrics1.loc['RMSE', 'Value'] = round(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold)),2)
read_across_metrics1.loc['RMSE/sigma', 'Value'] = round(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold))/Y_model.values.std(),2)
read_across_metrics1.loc['R2', 'Value'] = round(r2_score(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold),2)
read_across_metrics1.to_csv(path+'output/ReadAcrossMetrics1_%sReg.csv' %y_var)

print  "Cluster-based Read-across MAE: %f" %(sm.mean_absolute_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold)) 
print  "Cluster-based RMSE: %f" %np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold)) 
print  "Cluster-based RMSE/sigma: %f" %(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold))/Y_model.values.std())
print  "Cluster-based R2: %f" %(r2_score(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold))

#%%
#for n in range(1,6):
#    print n
#    print  "Cluster-based Read-across MAE: %f" %(sm.mean_absolute_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n])) 
#    print  "Cluster-based RMSE: %f" %np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n])) 
#    print  "Cluster-based RMSE/sigma: %f" %(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n]))/Y_model.values.std())
#    print  "Cluster-based R2: %f" %(r2_score(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n]))
read_across_metrics2 = pd.DataFrame(index = range(1,6), columns = {'MAE', 'RMSE', 'RMSE/sigma', 'R2'})
for n in range(1,6):
    read_across_metrics2.loc[n, 'MAE'] = round(sm.mean_absolute_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n]),2)
    read_across_metrics2.loc[n, 'RMSE'] = round(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n])),2)
    read_across_metrics2.loc[n, 'RMSE/sigma'] = round(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n]))/Y_model.values.std(),2)
    read_across_metrics2.loc[n, 'R2'] = round(r2_score(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n]),2)
    print n
    print  "Cluster-based Read-across MAE: %f" %(sm.mean_absolute_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n])) 
    print  "Cluster-based RMSE: %f" %np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n])) 
    print  "Cluster-based RMSE/sigma: %f" %(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n]))/Y_model.values.std())
    print  "Cluster-based R2: %f" %(r2_score(Y_model.loc[target_predicted_count.index], target_predicted_count['Clint%d' %n]))
read_across_metrics2.to_csv(path+'output/ReadAcrossMetrics2_%sReg.csv' %y_var)

#%%
#%%
###########################################################################
## Cluster-based regression (medium clearance)
###########################################################################

## Transform the data: Bin the clearance variable for classification
Y_clas = Y.copy()
[Y_clas.set_value(idx, int(-3)) for idx in Y_clas[Y_clas <= 0.9].index] 
[Y_clas.set_value(idx, int(-2)) for idx in Y_clas[(Y_clas > 0.9) & (Y_clas <= 50)].index] 
[Y_clas.set_value(idx, int(-1)) for idx in Y_clas[Y_clas > 50].index] 
Y_clas = pd.Series(Y_clas, index = Y.index) 

## Combine all the descriptors
data = pd.concat([Y_clas, fp], axis=1).dropna(axis=0, how='any')
X_model = data.ix[:, data.columns != y_var]
Y_model = data.ix[:, data.columns == y_var]
#%%
k = 20
target_predicted_count = pd.DataFrame(index = Y_model.index, columns = ['Clint Bin1', 'Clint Bin2','Clint Bin3','Clint Bin4','Clint Bin5'])
target_predicted_threshold = pd.DataFrame(index = Y_model.index, columns = ['Clint Bin'])

for target_idx in X_model.index:
    target = X_model.loc[target_idx,:]
    X_analog_pool = X_model[X_model.index != target_idx]
    
    # K-means clustering on the analog_pool data
    kmeans = KMeans(init='k-means++', n_clusters = k) 
    kmeans.fit(X_analog_pool) 
    k_means_labels = kmeans.labels_ 
    k_means_cluster_centers = kmeans.cluster_centers_ 
    k_means_labels_unique = np.unique(k_means_labels) 
    x_train_kmeans = kmeans.predict(X_analog_pool) 
    
    ## Create a new dataframe where each chemical is represented by a cluster number  
    ## and its associated Clearance value 
    df_kmeans = pd.DataFrame(x_train_kmeans,  index = X_analog_pool.index, columns = {'Cluster'}) 
    df_kmeans[y_var] = Y_model
    
    ## Assign target to a cluster
    target_cluster_no = int(kmeans.predict(target))
    analog_pool = df_kmeans[df_kmeans['Cluster'] == target_cluster_no].index.tolist()
    distances = dist.loc[target_idx, X_analog_pool.index].sort_values(ascending=False)
    
    # All analogs less than threshold
    analog_idx = distances[(distances > 0.70)].index.tolist()
    if len(analog_idx) == 0:
        pass
    else:
        pred = Counter([Y_clas[analog] for analog in analog_idx]).most_common(1)[0][0]
        target_predicted_threshold.loc[target_idx, 'Clint Bin'] = pred        

    # All analogs less than threshold and count
    for n in range(1,6):
        analog_idx = distances[(distances > 0.70)].index.tolist()[0:n]
        if len(analog_idx) == 0:
            pass
        else:
            pred = Counter([Y_clas[analog] for analog in analog_idx]).most_common(1)[0][0]
            target_predicted_count.loc[target_idx, 'Clint Bin%d' %n] = pred  
        
#%%
## Metrics by threshold
target_predicted_threshold = target_predicted_threshold.dropna()
Y_clas = Y_clas.loc[target_predicted_threshold.index]

true = [int(val) for val in Y_clas]
pred = [int(val) for val in target_predicted_threshold['Clint Bin']]

read_across_metrics1 = pd.DataFrame(columns = {'Value'})
read_across_metrics1.loc['Accuracy', 'Value'] = round(100*accuracy_score(true, pred),2)
read_across_metrics1.loc['F1-score', 'Value'] = [round(f1score,2) for f1score in f1_score(true, pred, average=None)]
read_across_metrics1.to_csv(path+'output/ReadAcrossMetrics1_%sClas.csv' %y_var)

print round(100*accuracy_score(true, pred),2)
print [round(f1score,2) for f1score in f1_score(true, pred, average=None)]

#%%
## Metrics by count and threshold
target_predicted_count = target_predicted_count.dropna()
Y_clas = Y_clas.loc[target_predicted_count.index]

true = [int(val) for val in Y_clas]

read_across_metrics2 = pd.DataFrame(columns = {'Accuracy', 'F1-score'})

for n in range(1,6):
    pred = [int(val) for val in target_predicted_count['Clint Bin%d' %n]]
    
    read_across_metrics2.loc[n, 'Accuracy'] = round(100*accuracy_score(true, pred),2)
    read_across_metrics2.loc[n, 'F1-score'] = [round(f1score,2) for f1score in f1_score(true, pred, average=None)]
    read_across_metrics2.to_csv(path+'output/ReadAcrossMetrics2_%sClas.csv' %y_var)
    
    print round(100*accuracy_score(true, pred),2)
    print [round(f1score,2) for f1score in f1_score(true, pred, average=None)]