# -*- coding: utf-8 -*- 
""" 
Created on Wed Aug 24 16:22:39 2016 
 
@author: ppradeep 
""" 
#%%
###########################################################################
## Import libraries
###########################################################################

import os
clear = lambda: os.system('cls')
clear()

## Import packages
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Classifiers
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn import svm
from sklearn.neural_network import MLPRegressor

# Machine learning relevant 
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as sm
from sklearn.metrics import r2_score

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
    scaler = preprocessing.StandardScaler().fit(X)
    transformed = scaler.transform(X)
    x_norm = pd.DataFrame(transformed, index = X.index) 
    x_norm.columns = X.columns
    return(x_norm)


def selectFeatures_perc(X, Y, percentile):
    model = SelectPercentile(f_classif, percentile)
    model = model.fit(X, Y) #convert datatype for use in the fit function
    scores = -np.log10(model.pvalues_)
    scores /= scores.max()
    X_tr = model.transform(X)
    ## Convert it into a dataframe 
    X_tr = pd.DataFrame(X_tr, index = X.index) 
    X_tr.columns = X.columns[model.get_support(indices=True)] 
    return X_tr

def selectFeatures_RFE(X, Y, n_features_to_select):
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select )
    rfe = rfe.fit(X, Y) #convert datatype for use in the fit function
    X_tr = rfe.transform(X)
    ## Convert it into a dataframe 
    X_tr = pd.DataFrame(X_tr, index = X.index) 
    X_tr.columns = X.columns[rfe.get_support(indices=True)] 
    return X_tr

   
def returnparams_knn(n_fold, X, Y):
    parameters = {'weights':['uniform', 'distance'], 'n_neighbors':[3,4,5], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    clf = KNeighborsRegressor()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    knn_params = grid_search.best_params_    
    return knn_params

def returnparams_lasso(n_fold, X, Y):
    parameters = {'alpha':[0.001, 0.05, 0.1, 1], 'tol': [0.01, 0.001], 'random_state':[5]}
    clf = Lasso()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    lasso_params = grid_search.best_params_
    return lasso_params

def returnparams_svm(n_fold, X, Y):
    parameters = {'kernel':['linear', 'rbf'], 'C':[0.1, 1, 10], 'gamma':[0.01, 0.1, 1], 'epsilon': [0.1, 1]}
    #parameters = {'kernel':['rbf'], 'C':[10], 'gamma':[0.01], 'epsilon': [0.1]}
    clf = svm.SVR()
    grid_search = GridSearchCV(clf, cv = 5, param_grid = parameters)
    grid_search.fit(X, Y)
    svm_params = grid_search.best_params_
    return svm_params
    
def returnparams_rf(n_fold, X, Y):
    parameters = {'n_estimators': [250, 500, 750, 1000], 'max_features': ['sqrt', 'auto'], 'random_state':[5]}
    #parameters = {'n_estimators': [1000], 'max_features': ['auto'], 'random_state':[5]}
    clf = RandomForestRegressor()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    rf_params = grid_search.best_params_
    return rf_params
    
def returnparams_gbr(n_fold, X, Y):
    parameters = {'n_estimators': [250, 500, 750], 'max_depth': [2,3,4], \
                  'random_state':[5], 'learning_rate': [0.01, 1], 'loss': ['ls', 'lad']}
    clf = GradientBoostingRegressor()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    gbr_params = grid_search.best_params_            
    return gbr_params
 
def returnparams_mlp(n_fold, X, Y): 
    parameters = {"solver": ['lbfgs', 'sgd', 'adam'], "activation": ['identity', 'logistic', 'tanh', 'relu'],\
                  'random_state':[5]}
    clf = MLPRegressor()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    mlp_params = grid_search.best_params_
    return mlp_params

def predict_y(clf, X, Y, n_fold):
    y = cross_val_predict(clf, X = X, y = Y, cv = n_fold)
    return y

def predict_test_y(clf, X, Y, X_test):
    clf = clf.fit(X,Y)
    y = clf.predict(X_test)
    return y

#%%
###########################################################################
## Set working directory 
###########################################################################
  
path = 'C:/Users/Administrator/OneDrive/Profile/Desktop/HTTK/'
#path = 'Z:/Projects/HTTK/' 

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


#%%
## HTTK package data
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

#%%
## Final Fub Data to Model
data.to_csv(path+'data/2-fub_data.csv', index_label = 'CASRN')

#%%
###########################################################################
## Read AR-ER data to keep those chemicals as an external test set
###########################################################################
#AR data
AR_data = pd.read_excel(path+'data/erar/data/Supplemental File 2_ARpathway_Results_ConfScores_CI_2016-08-30.xlsx', index_col='CASRN')
AR_ACC_columns = [col for col in AR_data if col.endswith('ACC')]
AR_data_subset = AR_data[(AR_data['AUC.Agonist']>0.1) | (AR_data['AUC.Antagonist']>0.1)][AR_ACC_columns]

#ER data
ER_data = pd.read_excel(path+'data/erar/data/S2 ER SuperMatrix 2015-03-24.xlsx', index_col='CASRN')
ER_ACC_columns = [col for col in ER_data if col.endswith('ACC')]
ER_data_subset = ER_data[(ER_data['AUC.Agonist']>0.1) | (ER_data['AUC.Antagonist']>0.1)][ER_ACC_columns]

## Combine ER-AR data
ERARdata = pd.concat([AR_data_subset, ER_data_subset], axis = 1)
ERARdata.replace(1000000, np.nan, inplace = True)

## Separate training data and external test data
trainingData = data.loc[data.index.difference(ERARdata.index)]
externaltestData = data.loc[ERARdata.index]

#%%
## Extract y data
Y = trainingData[y_var]
## Transform Y
#Y = Y[Y!= 0]
Y[Y==1.0] = 0.99
Y[Y==0] = 0.005

Y_model = (1-Y)/Y
Y_model = Y_model.apply(lambda x: np.log10(x))
Y_index = Y_model.index

## Histogram of transformed Y  
#plt.gcf().subplots_adjust(bottom=0.5)
#plt.figure(figsize=[12,8], dpi = 300)
#Y_model.hist(bins=20, alpha = 0.8, grid=False)
#plt.annotate('N = %d' %len(Y_model), [-4.5,160], size = 28)
#plt.xticks(fontsize = 24)
#plt.yticks(fontsize = 24)
#plt.xlabel('Transformed Fraction Unbound', size = 36, labelpad = 20)
#plt.ylabel('Frequency', size = 36, labelpad = 20)
#plt.savefig(path+'output/%sTrans_Hist.png'%y_var, bbox_inches='tight')

#%%
###########################################################################
## Read fingerprints and perform feature selection
###########################################################################

## Chemotyper FPs: 779 Toxprints 
df_chemotypes = pd.read_csv(path+'data/toxprint.txt', sep = ';', index_col='M_NAME') #Rename 'M_NAME' to 'CAS' in data file
## PubChem FPs: 881 bits
df_pubchem = pd.read_csv(path+'data/pubchem.txt', index_col='row ID')
# combine fingerprints
fingerprints = pd.concat([df_pubchem, df_chemotypes], axis=1)
# Remove culumns with >80% correlation
fingerprints = fingerprints.loc[Y_index,:].dropna()
fingerprints = selectFeatures_VarThresh(fingerprints, 0.80)
fingerprints = correlation(fingerprints, 0.80)

## Continuous descriptors
# OPERA 
df_opera = pd.read_csv(path+'data/OPERA2.4_Pred_QSARreadyStructures.csv', index_col='MoleculeID')[['LogP_pred','pKa_a_pred', 'pKa_b_pred']] #In MOE: Right click on mol -> Name -> Extract -> new field 'CAS'
df_opera['pKa_pred']=df_opera[['pKa_a_pred','pKa_b_pred']].min(axis=1)
opera = normalizeDescriptors(df_opera)

# PADEL descriptors
df_padel = pd.read_csv(path+'data/padel.txt', index_col='Name') 
df_padel = df_padel.loc[Y_index,:].dropna(axis=0, how='any') #drop columns that do not have PadDEL descriptors calculated
padel = normalizeDescriptors(df_padel)

# CDK descriptors
df_cdk = pd.read_csv(path+'data/cdk.txt', index_col='row ID') #Add CAS column to file
df_cdk = df_cdk.loc[Y_index,:].dropna(axis=0, how='any') #drop columns that do not have Y data or could not be calculated
cdk = normalizeDescriptors(df_cdk)

# Combine descriptors
descriptors = pd.concat([padel, cdk], axis=1).dropna()
# Drop correlated descriptors
descriptors = correlation(descriptors, 0.80)
# Select 10 descriptors
descriptors = selectFeatures_RFE(descriptors, Y.loc[descriptors.index], 10)
#descriptors = selectFeatures_perc(descriptors, Y_model.loc[descriptors.index], 1)

## Output file to capture the descriptors in the model for external predictions
features = pd.DataFrame({'Fingerprints': [fingerprints.columns.values.tolist()], 'opera': [opera.columns.values.tolist()], 'Padel+CDK': [descriptors.columns.values.tolist()]})
features.to_csv(path+'output/%s_Features.csv' %y_var)

#%%
###########################################################################
## Combine all the descriptors
###########################################################################
    
#1
#X_model = pd.concat([fingerprints], axis=1).dropna() #moe, descriptors, dft
#2
#X_model = pd.concat([fingerprints, opera[['LogP_pred', 'pKa_pred']]], axis=1).dropna() #moe, descriptors, dft
#3
X_model = pd.concat([fingerprints, opera[['LogP_pred', 'pKa_pred']], descriptors], axis=1).dropna() #moe, descriptors, dft

###########################################################################
##  Select the training and validation set
###########################################################################
index_random = X_model.index.values.tolist()
np.random.RandomState(40).shuffle(index_random) #set the seed to 40 to replicate results

n_idx = int(80*len(X_model)/100)
Y_train, Y_test = Y_model.ix[index_random[:n_idx]], Y_model.ix[index_random[n_idx:]]
X_train, X_test = X_model.ix[index_random[:n_idx]], X_model.ix[index_random[n_idx:]]

## Histogram of FINAL training and test data superimposed on each other
sigma_train = np.std(Y_train)
sigma_test = np.std(Y_test)

plt.figure(figsize=(8, 6), dpi = 200)
Y_train.hist(label = 'Training (n = %d | $\sigma$ = %0.2f)' %(len(Y_train), sigma_train), alpha = 0.75, color = 'r')
Y_test.hist(label = 'Test (n = %d| $\sigma$ = %0.2f)' %(len(Y_test), sigma_test), alpha = 0.75, color = 'g')
plt.xlabel('POD$_{tr}$', size = 24, labelpad = 10)
plt.ylabel('Frequency', size = 24, labelpad = 10)
plt.xticks(fontsize = 24)#, rotation = 90)
plt.yticks(fontsize = 24)
plt.legend(fontsize = 14, loc='upper left') 

plt.savefig(path+'/output/%s_TrainTestDist3.png' %y_var, bbox_inches='tight')
plt.show()

#%%
## Evaluate the hyper-parameters of each model
n_fold = 5

lasso_params = returnparams_lasso(n_fold, X_train, Y_train)
svm_params = returnparams_svm(n_fold, X_train, Y_train)
rf_params = returnparams_rf(n_fold, X_train, Y_train)
mlp_params = returnparams_mlp(n_fold, X_train, Y_train)

classifiers = [Lasso(**lasso_params),\
               svm.SVR(**svm_params),\
               RandomForestRegressor(**rf_params),\
               MLPRegressor(**mlp_params)     
               ]

## Make predictions
Y_predicted = pd.DataFrame(index = Y_train.index, columns = [str(clf).split('(')[0] for clf in classifiers])
Y_test_predicted = pd.DataFrame(index = Y_test.index, columns = [str(clf).split('(')[0] for clf in classifiers])

for clf in classifiers:
    # 5-fold internal cross-validation
    predicted = predict_y(clf, X_train, Y_train, n_fold)
    Y_predicted.loc[:,str(clf).split('(')[0]] = predicted
    # Fit model on entire training data and make predictions for test set
    predicted = predict_test_y(clf, X_train, Y_train, X_test)
    Y_test_predicted.loc[:,str(clf).split('(')[0]] = predicted


Y_predicted['Consensus (All)'] = Y_predicted.mean(axis = 1)
Y_test_predicted['Consensus (All)'] = Y_test_predicted.mean(axis = 1)

Y_predicted['Consensus (SVM,RF)'] = Y_predicted[['SVR', 'RandomForestRegressor']].mean(axis = 1)
Y_test_predicted['Consensus (SVM,RF)'] = Y_test_predicted[['SVR', 'RandomForestRegressor']].mean(axis = 1)

Y_predicted['Consensus (Lasso,RF)'] = Y_predicted[['Lasso', 'RandomForestRegressor']].mean(axis = 1)
Y_test_predicted['Consensus (Lasso,RF)'] = Y_test_predicted[['Lasso', 'RandomForestRegressor']].mean(axis = 1)

Y_predicted['Consensus (MLP,RF)'] = Y_predicted[['MLPRegressor', 'RandomForestRegressor']].mean(axis = 1)
Y_test_predicted['Consensus (MLP,RF)'] = Y_test_predicted[['MLPRegressor', 'RandomForestRegressor']].mean(axis = 1)

columns = ['MAE_int','RMSE_int', 'RMSE/sigma_int','R2_int', 'MAE_ext','RMSE_ext', 'RMSE/sigma_ext','R2_ext', 'params', 'coverage']

metrics = pd.DataFrame(index = Y_predicted.columns, columns = columns)
for key in Y_predicted:
    # save params
    if 'Lasso' in key:
        metrics.loc[key, 'params'] = [lasso_params]
    if 'SVR' in key:
        metrics.loc[key, 'params'] = [svm_params]
    if 'Random' in key:
        metrics.loc[key, 'params'] = [rf_params]
    if 'MLP' in key:
        metrics.loc[key, 'params'] = [mlp_params]
    # coverage 
    metrics.loc[key, 'coverage'] = [len(Y_predicted), len(Y_test_predicted)] #training, test
    # internal
    metrics.loc[key, 'MAE_int'] = round(sm.mean_absolute_error(Y_train, Y_predicted[key]),2)
    metrics.loc[key, 'RMSE_int'] = round(np.sqrt(sm.mean_squared_error(Y_train, Y_predicted[key])),2)
    metrics.loc[key, 'RMSE/sigma_int'] = round(np.sqrt(sm.mean_squared_error(Y_train, Y_predicted[key]))/np.std(Y_train),2)
    metrics.loc[key, 'R2_int'] = round(r2_score(Y_train, Y_predicted[key]),2)
    # external
    metrics.loc[key, 'MAE_ext'] = round(sm.mean_absolute_error(Y_test, Y_test_predicted[key]),2)
    metrics.loc[key, 'RMSE_ext'] = round(np.sqrt(sm.mean_squared_error(Y_test, Y_test_predicted[key])),2)
    metrics.loc[key, 'RMSE/sigma_ext'] = round(np.sqrt(sm.mean_squared_error(Y_test, Y_test_predicted[key]))/np.std(Y_test),2)
    metrics.loc[key, 'R2_ext'] = round(r2_score(Y_test, Y_test_predicted[key]),2)    

metrics.to_csv(path+'output/%s_Metrics3.csv' %y_var)                                

#%%
## Plot true versus predicted for the winning consensus models X selection #2
# Internal
plt.figure(figsize=[10,8], dpi = 300) #figsize=[12,8], dpi = 300

plt.plot(Y_train, Y_train, 'k', label = '')

# training set
plt.scatter(Y_train, Y_predicted['Consensus (SVM,RF)'], alpha = 0.3, color = 'r', s = 25, label = None)
plt.plot([Y_train.min(), Y_train.max()-sigma_train],[Y_train.min()+sigma_train, Y_train.max()],'r', label = '$\pm1 \sigma$(training) error interval', linestyle = '--')
plt.plot([Y_train.min(),Y_train.max()],[Y_train.min()-sigma_train, Y_train.max()-sigma_train],'r', linestyle = '--', label = None)

# PUT ERROR bar = 0.4 unit on Y_train['32385-11-8']
plt.errorbar(x = Y_train.ix['32385-11-8'], xerr = 0.4, y = Y_predicted['Consensus (SVM,RF)'].ix['32385-11-8']\
             ,fmt = 'o', ecolor = 'r', color = 'r', markersize='8', alpha=1, label = None)#, label = 'Observed Error')

# test set
plt.scatter(Y_test, Y_test_predicted['Consensus (SVM,RF)'], marker = 's', alpha = 0.3, color = 'b', s = 25, label = None)
plt.plot([Y_train.min(), Y_train.max()-sigma_test],[Y_train.min()+sigma_test, Y_train.max()],'b', label = '$\pm1 \sigma$(test) error interval', linestyle = ':')
plt.plot([Y_train.min(),Y_train.max()],[Y_train.min()-sigma_test, Y_train.max()-sigma_test],'b', linestyle = ':', label = None)
plt.xlim([Y_train.min(), Y_train.max()])
plt.ylim([Y_train.min(), Y_train.max()])

#training
plt.annotate('$RMSE (Training):$ %.2f' %metrics.loc['Consensus (SVM,RF)', 'RMSE_int'], [Y_train.min()+0.1, Y_train.max()-0.5], fontsize = 22)
plt.annotate('$R^{2} (Training):$ %.2f' %metrics.loc['Consensus (SVM,RF)', 'R2_int'], [Y_train.min()+0.1, Y_train.max()-1], fontsize = 22)
#test
plt.annotate('$RMSE (Test):$ %.2f' %metrics.loc['Consensus (SVM,RF)', 'RMSE_ext'], [Y_train.min()+0.1, Y_train.max()-1.75], fontsize = 22)
plt.annotate('$R^{2} (Test):$ %.2f' %metrics.loc['Consensus (SVM,RF)', 'R2_ext'], [Y_train.min()+0.1, Y_train.max()-2.25], fontsize = 22)

plt.legend(loc='lower right', numpoints = 2, scatterpoints = 1, fontsize = 15)
plt.xlabel('Observed', size = 36, labelpad = 20)
plt.ylabel('Predicted', size = 36, labelpad = 20)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)

plt.savefig(path+'/output/RF-SVM_TvsP_%s2.jpg' %(y_var), bbox_inches='tight')

#%%
