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


import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt

import sklearn.metrics as sm 
from sklearn.metrics import r2_score, accuracy_score, f1_score

from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Machine learning relevant 
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import  cross_validate, cross_val_predict, GridSearchCV
from sklearn import preprocessing
import sklearn.metrics as sm
from sklearn.metrics import r2_score
import itertools
from sklearn.metrics import confusion_matrix

#%%
###########################################################################
## User-defined functions
###########################################################################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 14)
    plt.yticks(tick_marks, classes, fontsize = 14)

#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], size = 38,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('Observed', fontsize = 40, labelpad = 20)
    plt.xlabel('Predicted', fontsize = 40, labelpad = 20)

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

   
def returnparams_lasso(n_fold, X, Y):
    parameters = {'alpha':[0.001, 0.05, 0.1, 1], 'tol': [0.01, 0.001], 'random_state':[5]}
    clf = Lasso()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    lasso_params = grid_search.best_params_
    print(lasso_params)
    return lasso_params

def returnparams_svm(n_fold, X, Y):
    parameters = {'kernel':['linear', 'rbf'], 'C':[0.001, 0.1, 1, 10], 'gamma':[0.01, 0.1, 1, 10], 'epsilon': [0.1, 1, 10]}
    clf = svm.SVR()
    grid_search = GridSearchCV(clf, cv = 5, param_grid = parameters)
    grid_search.fit(X, Y)
    svm_params = grid_search.best_params_
    print(svm_params)
    return svm_params
    
def returnparams_rf(n_fold, X, Y):
    parameters = {'n_estimators': [250, 500], 'max_features': ['sqrt', 'auto'], 'random_state':[5]}
    clf = RandomForestRegressor()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    rf_params = grid_search.best_params_
    print(rf_params)
    return rf_params
    
def returnparams_mlp(n_fold, X, Y): 
    parameters = {"solver": ['lbfgs', 'sgd', 'adam'], "activation": ['identity', 'logistic', 'tanh', 'relu'],\
                  'random_state':[5]}
    clf = MLPRegressor()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    mlp_params = grid_search.best_params_
    print(mlp_params)
    return mlp_params

def predict_y(clf, X, Y, n_fold):
    y = cross_val_predict(clf, X = X, y = Y, cv = n_fold)
    return y

def predict_test_y(clf, X, Y, X_test):
    clf = clf.fit(X,Y)
    y = clf.predict(X_test)
    return y

def returnparams_log(n_fold, X, Y): 
    parameters = {'C':[0.01, 0.05, 0.1, 1], 'tol': [0.1, 0.01, 0.001], 'solver':['newton-cg', 'lbfgs', 'sag'], 'random_state':[5]}
    clf = LogisticRegression()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    lr_params = grid_search.best_params_
    print(lr_params)
    return lr_params

def returnparams_svc(n_fold, X, Y):
    parameters = {'kernel':['linear', 'rbf'], 'C':[0.1, 1, 10], 'gamma':[0.01, 0.1, 1], 'decision_function_shape':['ovo', 'ovr']}
    clf = svm.SVC()
    grid_search = GridSearchCV(clf, cv = 5, param_grid = parameters)
    grid_search.fit(X, Y)
    svc_params = grid_search.best_params_
    print(svc_params)
    return svc_params
    
def returnparams_rfc(n_fold, X, Y):
    parameters = {"n_estimators": [50, 100, 250], "max_features": ['sqrt', 'log2', 'auto'], 'random_state':[5]}
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    rf_params = grid_search.best_params_
    print(rf_params)
    return rf_params

def returnparams_mlpc(n_fold, X, Y):
    parameters = {"solver": ['lbfgs', 'sgd', 'adam'], "activation": ['identity', 'logistic', 'tanh', 'relu'], 'random_state':[5]}
    clf = MLPClassifier()
    grid_search = GridSearchCV(clf, cv = n_fold, param_grid = parameters)
    grid_search.fit(X, Y)
    mlp_params = grid_search.best_params_
    print(mlp_params)
    return mlp_params

def calcMetrics(cnf_matrix):
    tn, fp, fn, tp = cnf_matrix.ravel()
    total = float(tp + tn + fp + fn)
    acc = round(100*float(tp + tn)/float(total),2)
    sens = round(100*float(tp)/float(tp + fp),2)
    spec = round(100*float(tn)/float(tn + fn),2)
    ba = round((sens+spec)/2,2)
    p_o = float(tp + tn)/total
    p_e = ((tp + fn)/total)*((tp + fp)/total) + ((fp + tn)/total)*((fn + tn)/total)
    kappa = round(((p_o - p_e)/(1 - p_e)), 2)
    return total, acc, sens, spec, ba, kappa

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

data.to_csv(path+'data/3-clint_data.csv', index_label = 'CASRN')

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
###########################################################################
### Read fingerprints and descriptors
###########################################################################
## Chemotyper FPs: 779 Toxprints 
df_chemotypes = pd.read_csv(path+'data/toxprint.txt', sep = ';', index_col='M_NAME') #Rename 'M_NAME' to 'CAS' in data file
## PubChem FPs: 881 bits
df_pubchem = pd.read_csv(path+'data/pubchem.txt', index_col='row ID')

## OPERA predictors
df_opera = pd.read_csv(path+'data/OPERA2.4_Pred_QSARreadyStructures.csv', index_col='MoleculeID')[['LogP_pred','pKa_a_pred', 'pKa_b_pred']] #In MOE: Right click on mol -> Name -> Extract -> new field 'CAS'
df_opera['pKa_pred']=df_opera[['pKa_a_pred','pKa_b_pred']].min(axis=1)
## PADEL descriptors
df_padel = pd.read_csv(path+'data/padel.txt', index_col='Name') 
## CDK descriptors
df_cdk = pd.read_csv(path+'data/cdk.txt', index_col='row ID') #Add CAS column to file

#%%
############################################################################
### CLASSIFICATION MODELS 
############################################################################
## Extract y data
Y = trainingData[y_var]

#%%
# Histogram of raw Y
#plt.gcf().subplots_adjust(bottom=0.5)
plt.figure()#figsize=[12,8], dpi = 300)
plt.hist(Y, bins = 10, alpha = 0.8)

plt.yticks(fontsize = 24)
plt.xlabel('Original Intrinsic Clearance', size = 36, labelpad = 20)
plt.ylabel('Frequency', size = 36, labelpad = 20)
plt.savefig(path+'output/%s_Hist.png'%y_var, bbox_inches='tight')

#%%
## Transform the data: Bin the clearance variable for classification
Y_clas = Y.copy()
[Y_clas.set_value(idx, int(-3)) for idx in Y_clas[Y_clas <= 0.9].index] 
[Y_clas.set_value(idx, int(-2)) for idx in Y_clas[(Y_clas > 0.9) & (Y_clas <= 50)].index] 
[Y_clas.set_value(idx, int(-1)) for idx in Y_clas[Y_clas > 50].index] 
Y_clas = pd.Series(Y_clas, index = Y.index) 
# Histogram of transformed Y  
plt.gcf().subplots_adjust(bottom=0.5)
plt.figure(figsize=[12,8], dpi = 300)
Y_clas.hist(bins=3, alpha = 0.8, color = 'r')
labels = ['Low', 'Medium', 'High']
plt.xticks([-3, -2, -1], labels, size = 18)
plt.xlabel('Transformed Clearance (for clustering)', size = 36, labelpad = 20)
plt.ylabel('Frequency', size = 36, labelpad = 20)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.tight_layout()
plt.savefig(path+'output/%sTrans_Hist.png'%y_var, bbox_inches='tight')

#%%
############################################################################
### Perform feature selection
############################################################################
# Fingerprints
# combine fingerprints
fingerprints = pd.concat([df_pubchem, df_chemotypes], axis=1).dropna()
fingerprints = fingerprints.loc[Y_clas.index, :].dropna()
fingerprints = selectFeatures_VarThresh(fingerprints, 0.80)
fingerprints = correlation(fingerprints, 0.80)

## Continuous descriptors
opera = normalizeDescriptors(df_opera)
padel = df_padel.loc[Y_clas.index,:].dropna(axis=0, how='any') #drop columns that do not have PadDEL descriptors calculated
padel = normalizeDescriptors(padel)
cdk = df_cdk.loc[Y_clas.index,:].dropna(axis=0, how='any') #drop columns that do not have Y data or could not be calculated
cdk = normalizeDescriptors(cdk)

#%%
## Combine descriptors
descriptors = pd.concat([padel, cdk], axis=1).dropna()
# Drop correlated descriptors
descriptors = correlation(descriptors, 0.80)
# Select 10 descriptors
descriptors = selectFeatures_RFE(descriptors, Y_clas.loc[descriptors.index], 10)
#descriptors = selectFeatures_perc(descriptors, Y_clas.loc[descriptors.index], 1)
#%%
## Output file to capture the descriptors in the model for external predictions
clint_features = pd.DataFrame({'Fingerprints': [fingerprints.columns.values.tolist()], 'opera': [opera.columns.values.tolist()], 'Padel+CDK': [descriptors.columns.values.tolist()]})
clint_features.to_csv(path+'output/Clint_Features_Classification.csv')

#%%
############################################################################
###  Set X data for modeling
############################################################################
#1
X_model = pd.concat([fingerprints], axis=1).dropna() #, moe, descriptors, dft
#2
X_model = pd.concat([fingerprints, opera[['LogP_pred', 'pKa_pred']]], axis=1).dropna() #, moe, descriptors, dft
#3
X_model = pd.concat([fingerprints, opera[['LogP_pred', 'pKa_pred']], descriptors], axis=1).dropna() #, moe, descriptors, dft

############################################################################
###  Set the training and validation set for classification model
############################################################################
index_random = X_model.index.values.tolist()
np.random.RandomState(40).shuffle(index_random) #set the seed to 40 to replicate results

n_idx = int(80*len(X_model)/100)
X_train, X_test = X_model.ix[index_random[:n_idx]], X_model.ix[index_random[n_idx:]]
Y_train, Y_test = Y_clas.ix[index_random[:n_idx]], Y_clas.ix[index_random[n_idx:]]

###########################################################################
## Build the classification models
###########################################################################

## Evaluate the hyper-parameters of each model
n_fold = 5

lasso_params = returnparams_log(n_fold, X_train, Y_train.values.tolist())
svm_params = returnparams_svc(n_fold, X_train, Y_train.values.tolist())
rf_params = returnparams_rfc(n_fold, X_train, Y_train.values.tolist())
mlp_params = returnparams_mlpc(n_fold, X_train, Y_train.values.tolist())

classifiers = [LogisticRegression(**lasso_params),\
               svm.SVC(**svm_params),\
               RandomForestClassifier(**rf_params),\
               MLPClassifier(**mlp_params)     
               ]

## Make predictions
Y_predicted = pd.DataFrame(index = Y_train.index, columns = [str(clf).split('(')[0] for clf in classifiers])
Y_test_predicted = pd.DataFrame(index = Y_test.index, columns = [str(clf).split('(')[0] for clf in classifiers])

for clf in classifiers:
    # 5-fold internal cross-validation
    predicted = predict_y(clf, X_train, Y_train.values.tolist(), n_fold)
    Y_predicted.loc[:,str(clf).split('(')[0]] = predicted
    
    # Fit model on entire training data and make predictions for test set
    predicted = predict_test_y(clf, X_train, Y_train.values.tolist(), X_test)
    Y_test_predicted.loc[:,str(clf).split('(')[0]] = predicted

Y_predicted['Consensus'] = Y_predicted.mean(axis = 1).apply(np.round)
Y_test_predicted['Consensus'] = Y_test_predicted.mean(axis = 1).apply(np.round)

Y_predicted['Consensus (SVM,RF)'] = Y_predicted[['SVC', 'RandomForestClassifier']].mean(axis = 1).apply(np.round)
Y_test_predicted['Consensus (SVM,RF)'] = Y_test_predicted[['SVC', 'RandomForestClassifier']].mean(axis = 1).apply(np.round)

columns = ['Accuracy_int','F1score_int', 'Accuracy_ext','F1score_ext', 'params', 'coverage']
metrics = pd.DataFrame(index = Y_predicted.columns, columns = columns)

for key in Y_predicted:
    # save params
    if 'Log' in key:
        metrics.loc[key, 'params'] = [lasso_params]
    if 'SVC' in key:
        metrics.loc[key, 'params'] = [svm_params]
    if 'Random' in key:
        metrics.loc[key, 'params'] = [rf_params]
    if 'MLP' in key:
        metrics.loc[key, 'params'] = [mlp_params]
    # coverage 
    metrics.loc[key, 'coverage'] = [len(Y_predicted), len(Y_test_predicted)] #training, test
    # internal
    cnf_matrix = confusion_matrix(Y_train.values.tolist(), Y_predicted[key])
    metrics.loc[key, 'Accuracy_int'] = round(100*accuracy_score(Y_train.values.tolist(), Y_predicted[key]),2)
    metrics.loc[key, 'F1score_int'] = [round(f1score,2) for f1score in f1_score(Y_train.values.tolist(), Y_predicted[key], average=None)]
    # external
    cnf_matrix = confusion_matrix(Y_test.values.tolist(), Y_test_predicted[key])
    metrics.loc[key, 'Accuracy_ext'] = round(100*accuracy_score(Y_test.values.tolist(), Y_test_predicted[key]),2)
    metrics.loc[key, 'F1score_ext'] = [round(f1score,2) for f1score in f1_score(Y_test.values.tolist(), Y_test_predicted[key], average=None)]
  
metrics.to_csv(path+'output/Clint-clas_Metrics3.csv')   

#%%
############################################################################
### REGRESSION MODELS 
############################################################################
## Extract y data for regression
Y_reg = Y[(Y > 0.9) & (Y <= 50)]
# Histogram of Medium Y
plt.gcf().subplots_adjust(bottom=0.5)
plt.figure(figsize=[12,8], dpi = 300)
Y_reg.hist(bins=20, alpha = 0.8)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.xlabel('Original Clearance (Medium)', size = 36, labelpad = 20)
plt.ylabel('Frequency', size = 36, labelpad = 20)
plt.tight_layout()
plt.savefig(path+'output/%sMedium_Hist.png'%y_var)

## Transform Y
Y_reg = Y_reg.apply(lambda x: np.log10(x))
# Histogram of transformed Y  
plt.gcf().subplots_adjust(bottom=0.5)
plt.figure(figsize=[12,8], dpi = 200)
Y_reg.hist(bins=20, alpha = 0.8, color='red')
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.xlabel('Transformed Clearance (for regression)', size = 36, labelpad = 20)
plt.ylabel('Frequency', size = 36, labelpad = 20)
plt.tight_layout()
plt.savefig(path+'output/%sMediumTrans_Hist.png'%y_var)


#%%
############################################################################
### Perform feature selection
############################################################################
# Fingerprints
# combine fingerprints
fingerprints = pd.concat([df_pubchem, df_chemotypes], axis=1).dropna()
fingerprints = fingerprints.loc[Y_reg.index, :].dropna()
fingerprints = selectFeatures_VarThresh(fingerprints, 0.80)
fingerprints = correlation(fingerprints, 0.80)

# Continuous descriptors
# OPERA
opera = normalizeDescriptors(df_opera)
# PADEL descriptors
padel = df_padel.loc[Y_reg.index,:].dropna(axis=0, how='any') #drop columns that do not have PadDEL descriptors calculated
padel = normalizeDescriptors(padel)
# CDK descriptors
cdk = df_cdk.loc[Y_reg.index,:].dropna(axis=0, how='any') #drop columns that do not have Y data or could not be calculated
cdk = normalizeDescriptors(cdk)

## Combine descriptors
descriptors = pd.concat([padel, cdk], axis=1).dropna()
# Drop correlated descriptors
descriptors = correlation(descriptors, 0.80)

# Select 10 descriptors
descriptors1 = descriptors.loc[Y_reg.index].dropna()
descriptors = selectFeatures_RFE(descriptors1, Y_reg.loc[descriptors1.index], 10)
#descriptors = selectFeatures_perc(descriptors, Y_reg.loc[descriptors.index], 1)

#%%
## Output file to capture the descriptors in the model for external predictions
clint_features = pd.DataFrame({'Fingerprints': [fingerprints.columns.values.tolist()], 'opera': [opera.columns.values.tolist()], 'Padel+CDK': [descriptors.columns.values.tolist()]})
clint_features.to_csv(path+'output/Clint_Features_Regression.csv')

#%%
############################################################################
###  Set X data for modeling by adding more descriptors
############################################################################
#1
X_model = pd.concat([fingerprints], axis=1).loc[Y_reg.index].dropna() #, moe, descriptors, dft
#2
#X_model = pd.concat([fingerprints, opera[['LogP_pred', 'pKa_pred']]], axis=1).loc[Y_reg.index].dropna()#, moe, descriptors, dft
#3
#X_model = pd.concat([fingerprints, opera[['LogP_pred', 'pKa_pred']], descriptors], axis=1).loc[Y_reg.index].dropna() #, moe, descriptors, dft

###########################################################################
##  Set the training and validation set for regression model
###########################################################################
index_random = X_model.index.values.tolist()
np.random.RandomState(210).shuffle(index_random) #set the seed to 40 to replicate results

n_idx = int(80*len(X_model)/100)
Y_train, Y_test = Y_reg.ix[index_random[:n_idx]], Y_reg.ix[index_random[n_idx:]]
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

plt.savefig(path+'/output/%s_TrainTestDist1.png' %y_var, bbox_inches='tight')
plt.show()


###########################################################################
##  Build the regression model
###########################################################################
## Evaluate the hyper-parameters of each model
n_fold = 5
#
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

#Y_predicted['Consensus'] = Y_predicted.mean(axis = 1)
#Y_test_predicted['Consensus'] = Y_test_predicted.mean(axis = 1)
#
Y_predicted['Consensus (SVM,RF)'] = Y_predicted[['SVR', 'RandomForestRegressor']].mean(axis = 1)
Y_test_predicted['Consensus (SVM,RF)'] = Y_test_predicted[['SVR', 'RandomForestRegressor']].mean(axis = 1)

Y_predicted['Consensus (SVM,Lasso)'] = Y_predicted[['SVR', 'Lasso']].mean(axis = 1)
Y_test_predicted['Consensus (SVM,Lasso)'] = Y_test_predicted[['SVR', 'Lasso']].mean(axis = 1)

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
metrics.to_csv(path+'output/Clint-Reg_Metrics1.csv')                        

#%%
## Plot true versus predicted for the consensus models

plt.figure(figsize=[10,8], dpi = 300)

# Training
plt.scatter(Y_train, Y_predicted['RandomForestRegressor'], alpha = 0.4, color = 'r', s = 25, label = None)
plt.plot([Y_train.min(), Y_train.max()-sigma_train],[Y_train.min()+sigma_train, Y_train.max()],'r', label = '$\pm1 \sigma$ error interval', linestyle = '--')
plt.plot([Y_train.min(),Y_train.max()],[Y_train.min()-sigma_train, Y_train.max()-sigma_train],'r', linestyle = '--', label = None)

# Test
plt.scatter(Y_test, Y_test_predicted['RandomForestRegressor'], marker = 's', alpha = 0.4, color = 'b', s = 25, label = None)
plt.plot([Y_train.min(), Y_train.max()-sigma_test],[Y_train.min()+sigma_test, Y_train.max()],'b', label = '$\pm1 \sigma$ error interval', linestyle = '--')
plt.plot([Y_train.min(),Y_train.max()],[Y_train.min()-sigma_test, Y_train.max()-sigma_test],'b', linestyle = '--', label = None)
#
# PUT ERROR bar = 0.4 unit on Y_train[0]
plt.errorbar(x = Y_train.ix['119-90-4'], xerr = 0.3, y = Y_predicted['RandomForestRegressor'].ix['119-90-4']\
             ,fmt = 'o', ecolor = 'r', color = 'r', markersize='8', alpha=1)#, label = 'Observed Error')

plt.plot(Y_train, Y_train, 'k', label = '')

plt.xlim([Y_train.min(), Y_train.max()])
plt.ylim([Y_train.min(), Y_train.max()])

#training
plt.annotate('$RMSE (Test): %.2f$' %metrics.loc['RandomForestRegressor', 'RMSE_ext'], [0.98, 0.125], fontsize = 21)
plt.annotate('$R^{2} (Test): %.2f$' %metrics.loc['RandomForestRegressor', 'R2_ext'], [0.98, 0.025], fontsize = 21)
#test
plt.annotate('$RMSE (Training): %.2f$' %metrics.loc['RandomForestRegressor', 'RMSE_int'], [0.98, 0.37], fontsize = 21)
plt.annotate('$R^{2} (Training): %.2f$' %metrics.loc['RandomForestRegressor', 'R2_int'], [0.98, 0.27], fontsize = 21)

plt.legend(loc='upper left', numpoints = 1, scatterpoints = 1, fontsize = 15)

plt.xlabel('Observed', size = 36, labelpad = 20)
plt.ylabel('Predicted', size = 36, labelpad = 20)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.savefig(path+'/output/RF_TvsP-1_%s.jpg' %y_var, bbox_inches='tight')
#plt.savefig(path+'/output/Int_Consensus_TvsP_%s_%s1.jpg' %(str(clf).split('(')[0], y_var), bbox_inches='tight')

#%%