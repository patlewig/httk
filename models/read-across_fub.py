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
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold 

import sklearn.metrics as sm 
from sklearn import cross_validation
from sklearn.metrics import r2_score
from sklearn import preprocessing

from sklearn.cluster import KMeans 
#%%
## User-defined functions
def selectFeatures_VarThresh(X, threshold):
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold))) 
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
data1 = pd.read_csv(path+'data/Prachi-112117.txt', index_col = 'CAS')#.loc[:,['All.Compound.Names', 'Human.Funbound.plasma', 'Human.Clint']]
#%%
data1.rename(columns={'All.Compound.Names' : 'Name'}, inplace = True)
data2 = pd.read_excel(path+'data/AFFINITY_Model_Results-2018-02-27.xlsx', index_col = 'CAS').loc[:,['Name','Fup.Med']]
data2.rename(columns={'Name': 'All.Compound.Names','Fup.Med':'Human.Funbound.plasma'}, inplace = True)
data3 = pd.read_excel(path+'data/CLint-2018-03-01-Results.xlsx', index_col = 'CAS').loc[:,['Name','CLint.1uM.Median']]
data3.rename(columns={'Name': 'All.Compound.Names','CLint.1uM.Median':'Human.Clint'}, inplace = True)

#%%
# Set y variable
y_var = 'Human.Funbound.plasma'
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
#%%
###########################################################################
## Set data for modeling
###########################################################################
## Transform Y
Y[Y==1.0] = 0.99
Y[Y==0] = 0.005

Y_model = (1-Y)/Y
Y_model = Y_model.apply(lambda x: np.log10(x))
Y_index = Y_model.index

#%%
###########################################################################
## Read fingerprints
###########################################################################

## PubChem FPs: 881 bits
## Chemotyper FPs: 779 Toxprints 
fp = pd.read_csv(path+'data/CombinedFP.csv', index_col='row ID') 
### PubChem FPs: 881 bits
#df_pubchem = pd.read_csv(path+'data/pubchem.txt', index_col='row ID')
## Combined fingerprints
#fp = pd.concat([df_chemotypes, df_pubchem], axis = 1)
## Remove culumns with >80% correlation
#fp = fp.loc[Y_index,:].dropna()
#fp = selectFeatures_VarThresh(fp, 0.80)
#fp = correlation(fp, 0.80)

#%%
###########################################################################
## Combine all the descriptors
###########################################################################
data = pd.concat([Y_model, fp], axis=1).dropna(axis=0, how='any')
X_model = data.ix[:, data.columns != y_var]
Y_model = data.ix[:, data.columns == y_var]

#%%
## PubChem distances 
dist = pd.read_csv(path+'data/CombinedFPDistances.csv', index_col='row ID') 
dist.columns = dist.index 

#%%
###########################################################################
## Simple read-across (no clustering)
###########################################################################
k = 30 #from unsupervised clustering results. Clint: 25, Fub: 30
target_predicted_count = pd.DataFrame(index = Y_model.index, columns = ['Predicted Fub1', 'Predicted Fub2','Predicted Fub3','Predicted Fub4','Predicted Fub5'])
target_predicted_threshold = pd.DataFrame(index = Y_model.index, columns = ['Predicted Fub'])

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
        target_predicted_threshold.loc[target_idx, 'Predicted Fub'] = pred        

#    # All analogs less than threshold and count
#    for n in range(1,6):
#        analog_idx = distances[(distances > 0.70)].index.tolist()[0:n]
#        if len(analog_idx) == 0:
#            pass
#        else:
#            pred = np.average([Y_model.loc[analog] for analog in analog_idx])
#            target_predicted_count.loc[target_idx, 'Predicted Fub%d' %n] = pred  
   
#%%
target_predicted_threshold = target_predicted_threshold.dropna()

read_across_metrics1 = pd.DataFrame(columns = {'Value'})
read_across_metrics1.loc['MAE', 'Value'] = round(sm.mean_absolute_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold),2)
read_across_metrics1.loc['RMSE', 'Value'] = round(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold)),2)
read_across_metrics1.loc['RMSE/sigma', 'Value'] = round(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold))/Y_model.values.std(),2)
read_across_metrics1.loc['R2', 'Value'] = round(r2_score(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold),2)
read_across_metrics1.to_csv(path+'output/ReadAcrossMetrics1_%s.csv' %y_var)

print  "Cluster-based Read-across MAE: %f" %(sm.mean_absolute_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold)) 
print  "Cluster-based RMSE: %f" %np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold)) 
print  "Cluster-based RMSE/sigma: %f" %(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold))/Y_model.values.std())
print  "Cluster-based R2: %f" %(r2_score(Y_model.loc[target_predicted_threshold.index], target_predicted_threshold))

#%%
target_predicted_count = target_predicted_count.dropna()

read_across_metrics2 = pd.DataFrame(index = range(1,6), columns = {'MAE', 'RMSE', 'RMSE/sigma', 'R2'})
for n in range(1,6):
    read_across_metrics2.loc[n, 'MAE'] = round(sm.mean_absolute_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Predicted Fub%d' %n]),2)
    read_across_metrics2.loc[n, 'RMSE'] = round(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Predicted Fub%d' %n])),2)
    read_across_metrics2.loc[n, 'RMSE/sigma'] = round(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Predicted Fub%d' %n]))/Y_model.values.std(),2)
    read_across_metrics2.loc[n, 'R2'] = round(r2_score(Y_model.loc[target_predicted_count.index], target_predicted_count['Predicted Fub%d' %n]),2)
    print n
    print  "Cluster-based Read-across MAE: %f" %(sm.mean_absolute_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Predicted Fub%d' %n])) 
    print  "Cluster-based RMSE: %f" %np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Predicted Fub%d' %n])) 
    print  "Cluster-based RMSE/sigma: %f" %(np.sqrt(sm.mean_squared_error(Y_model.loc[target_predicted_count.index], target_predicted_count['Predicted Fub%d' %n]))/Y_model.values.std())
    print  "Cluster-based R2: %f" %(r2_score(Y_model.loc[target_predicted_count.index], target_predicted_count['Predicted Fub%d' %n]))
read_across_metrics2.to_csv(path+'output/ReadAcrossMetrics2_%s.csv' %y_var)

 
