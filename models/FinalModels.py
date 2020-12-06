# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:37:03 2019

@author: ppradeep
"""
import os
clear = lambda: os.system('cls')
clear()

## Import packages
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle

# Classifiers
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm, preprocessing

path = 'C:/Users/Administrator/OneDrive/Profile/Desktop/HTTK/'
#path = 'Z:/Projects/HTTK/' 

#%%
# Normalize descriptors: Transform variables to mean=0, variance=1
def normalizeDescriptors(X):
    scaler = preprocessing.StandardScaler().fit(X)
    transformed = scaler.transform(X)
    x_norm = pd.DataFrame(transformed, index = X.index) 
    x_norm.columns = X.columns
    return(scaler, x_norm)
    
#%%
###########################################################################
###########################################################################
## Build the final models
###########################################################################
###########################################################################

####-----------------------------------------------------------------------------------------------------------------
## Read training data 
####-----------------------------------------------------------------------------------------------------------------
data1 = pd.read_csv(path+'data/Prachi-112117.txt', index_col = 'CAS').loc[:,['All.Compound.Names', 'Human.Funbound.plasma', 'Human.Clint']]
data1.rename(columns={'All.Compound.Names' : 'Name'}, inplace = True)
data2 = pd.read_excel(path+'data/AFFINITY_Model_Results-2018-02-27.xlsx', index_col = 'CAS').loc[:,['Name','Fup.Med']]
data2.rename(columns={'Name': 'All.Compound.Names','Fup.Med':'Human.Funbound.plasma'}, inplace = True)
data3 = pd.read_excel(path+'data/CLint-2018-03-01-Results.xlsx', index_col = 'CAS').loc[:,['Name','CLint.1uM.Median']]
data3.rename(columns={'Name': 'All.Compound.Names','CLint.1uM.Median':'Human.Clint'}, inplace = True)

#%%
####-----------------------------------------------------------------------------------------------------------------
## Read training fingerprints
####-----------------------------------------------------------------------------------------------------------------
## Chemotyper FPs: 779 Toxprints 
df_chemotypes = pd.read_csv(path+'data/toxprint.txt', sep = ';', index_col='M_NAME') #Rename 'M_NAME' to 'CAS' in data file
## PubChem FPs: 881 bits
df_pubchem = pd.read_csv(path+'data/pubchem.txt', index_col='row ID')

####-----------------------------------------------------------------------------------------------------------------
## Read continuous descriptors 
####-----------------------------------------------------------------------------------------------------------------
### OPERA descriptors
df_opera = pd.read_csv(path+'data/OPERA2.5_Pred.csv', index_col='MoleculeID')[['LogP_pred','pKa_a_pred', 'pKa_b_pred']] #In MOE: Right click on mol -> Name -> Extract -> new field 'CAS'
df_opera['pKa_pred']=df_opera[['pKa_a_pred','pKa_b_pred']].min(axis=1)
opera_scaler, opera = normalizeDescriptors(df_opera)#[['pKa_pred','LogP_pred']]
opera = opera[['pKa_pred','LogP_pred']]
## PADEL descriptors
df_padel = pd.read_csv(path+'data/padel.txt', index_col='Name').dropna() 
padel_scaler, padel = normalizeDescriptors(df_padel)
## CDK descriptors
df_cdk = pd.read_csv(path+'data/cdk.txt', index_col='row ID').dropna() #Add CAS column to file
cdk_scaler, cdk = normalizeDescriptors(df_cdk)

#%%
####-----------------------------------------------------------------------------------------------------------------
## Save the normalization vector 
####-----------------------------------------------------------------------------------------------------------------
pickle.dump(opera_scaler, open(path+'output/opera_scaler.sav', 'wb'))
pickle.dump(padel_scaler, open(path+'output/padel_scaler.sav', 'wb'))
pickle.dump(cdk_scaler, open(path+'output/cdk_scaler.sav', 'wb'))
#%%
####-----------------------------------------------------------------------------------------------------------------
## Features from the 5-fold CV model
####-----------------------------------------------------------------------------------------------------------------
fub_features = pd.read_csv(path+'output/Human.Funbound.plasma_Features.csv')
clint_features_clas = pd.read_csv(path+'output/Clint_Features_Classification.csv')
clint_features_reg = pd.read_csv(path+'output/Clint_Features_Regression.csv')

#%%
####-----------------------------------------------------------------------------------------------------------------
## Model for Fraction Unbound in Plasma
####-----------------------------------------------------------------------------------------------------------------
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
####-----------------------------------------------------------------------------------------------------------------
## Extract y data
####-----------------------------------------------------------------------------------------------------------------
Y = data[y_var]
## Set data for modeling
## Transform Y
Y[Y==1.0] = 0.99
Y[Y==0] = 0.005
Y_model = (1-Y)/Y
Y_model = Y_model.apply(lambda x: np.log10(x))
Y_index = Y_model.index

#%%
####-----------------------------------------------------------------------------------------------------------------
## Combine fingerprints 
####-----------------------------------------------------------------------------------------------------------------
fingerprints = pd.concat([df_pubchem, df_chemotypes], axis=1).dropna()
fingerprints = fingerprints.loc[Y_index,:].dropna()
# Select fingerprints from the feature file
retain = [str(val.replace("'", "").replace(" ", "")) for val in fub_features.ix[0,'Fingerprints'].split(',')]
retain[0] = retain[0].replace("[", "")
retain[len(retain)-1] = retain[len(retain)-1].replace("c]",'c') ##manually check the last entry and correct it
fingerprints = fingerprints.loc[:,retain]
####-----------------------------------------------------------------------------------------------------------------
## Combine descriptors
####-----------------------------------------------------------------------------------------------------------------
descriptors = pd.concat([padel, cdk], axis=1).dropna()
descriptors = descriptors.loc[Y_index,:].dropna()
# Select descriptors from the feature file
retain = [str(val.replace("'", "").replace(" ", "")) for val in fub_features.ix[0,'Padel+CDK'].split(',')]
retain[0] = retain[0].replace("[", "")
retain[len(retain)-1] = retain[len(retain)-1].replace("]",'')
descriptors = descriptors.loc[:,retain]
####-----------------------------------------------------------------------------------------------------------------
## Combine all the descriptors and set the X and Y for training the model
####-----------------------------------------------------------------------------------------------------------------
data = pd.concat([Y_model, fingerprints, opera], axis=1).dropna(axis=0, how='any')
X_fub_model = data.ix[:, data.columns != y_var]
Y_fub_model = data[y_var]

meanY = np.mean(Y_fub_model)
stdY = np.std(Y_fub_model)

#%%
## Histogram of the final training set
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6), dpi = 300)

Y_fub_model.hist(alpha = 0.75, color = 'r', grid = False)
plt.annotate('N = %d' %len(Y_fub_model), [-2.5,200], size = 20)
plt.annotate('$\mu = %0.2f$' %(meanY), [-2.5,185], size = 20)
plt.annotate('$\sigma = %0.2f$' %(stdY), [-2.5,170], size = 20)

plt.xlabel('Fub$_{tr}$', size = 24, labelpad = 10)
plt.ylabel('Frequency', size = 24, labelpad = 10)
plt.xticks(fontsize = 24)#, rotation = 90)
plt.yticks(fontsize = 24)

plt.savefig(path+'/output/%s_TrainingData.png' %y_var, bbox_inches='tight')
plt.show()

data.to_csv(path+'output/fub_trainingdata.csv', index_label='CASRN')
#%%
####-----------------------------------------------------------------------------------------------------------------
## Develop model 

clf_fub1 = svm.SVR(epsilon = 0.1, C = 10, gamma = 0.01, kernel = "rbf")
clf_fub1 = clf_fub1.fit(X = X_fub_model, y = Y_fub_model)

clf_fub2 = RandomForestRegressor(max_features = 'auto', n_estimators = 1000, random_state = 5)
clf_fub2 = clf_fub2.fit(X = X_fub_model, y = Y_fub_model)
#
## Save the models to disk
pickle.dump(clf_fub1, open(path+'output/fub_svr.sav', 'wb'))
pickle.dump(clf_fub2, open(path+'output/fub_rf.sav', 'wb'))


#%%
###########################################################################
## Models for Intrinsic Clearance 
###########################################################################

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

#%%
data = pd.DataFrame(index = casList, columns = ['Name',y_var])

#%%
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
## Transform the data: Bin the clearance variable for classification
Y = data[y_var]
Y_clas = Y.copy()
[Y_clas.set_value(idx, int(-3)) for idx in Y_clas[Y_clas <= 0.9].index] 
[Y_clas.set_value(idx, int(-2)) for idx in Y_clas[(Y_clas > 0.9) & (Y_clas <= 50)].index] 
[Y_clas.set_value(idx, int(-1)) for idx in Y_clas[Y_clas > 50].index] 
Y_clas = pd.Series(Y_clas, index = Y.index) 

low_median = Y[Y_clas[Y_clas==-3].index].median()
high_median = Y[Y_clas[Y_clas==-1].index].median()

###########################################################################
## Classification:

## Combine fingerprints and perform feature selection
fingerprints = pd.concat([df_pubchem, df_chemotypes], axis=1).dropna()
fingerprints = fingerprints.loc[Y_clas.index,:].dropna()
# Select fingerprints from the feature file
retain = [str(val.replace("'", "").replace(" ", "")) for val in clint_features_clas.ix[0,'Fingerprints'].split(',')]
retain[0] = retain[0].replace("[", "")
retain[len(retain)-1] = retain[len(retain)-1].replace("]",'')
fingerprints = fingerprints.loc[:,retain]

#%%
## Classification: Combine all the descriptors and set the X and Y for training the classification model
data = pd.concat([Y_clas, fingerprints, opera], axis=1).dropna(axis=0, how='any')
X_ClintClas_model = data.ix[:, data.columns != y_var]
Y_ClintClas_model = data[y_var]

#%%
data.to_csv(path+'output/clintclas_trainingdata.csv', index_label='CASRN')

#%%
## Histogram of the final training set
import matplotlib.pyplot as plt
plt.gcf().subplots_adjust(bottom=0.5)
plt.figure(figsize=[8,6], dpi = 300)
plt.hist(Y_ClintClas_model.values.tolist(), color = 'r', align = 'left', rwidth = 1)
plt.annotate('N = %d' %len(Y_ClintClas_model.values.tolist()), [-3.15,260], size = 24)

labels = ['Low', 'Medium', 'High']
plt.xticks([-3, -2, -1], labels, size = 18)
plt.xlabel('Transformed Clearance \n(for classification)', size = 28, labelpad = 5)
plt.ylabel('Frequency', size = 28, labelpad = 5)
plt.xticks(fontsize = 20)#, rotation = 90)
plt.yticks(fontsize = 20)
plt.savefig(path+'output/%sClas_TrainingData.png'%y_var, bbox_inches='tight')

#%%
###########################################################################
## Develop classification model 

## Classification model
clf_clintClas = svm.SVC(C=10, decision_function_shape='ovo', gamma=0.01, kernel='rbf')
clf_clintClas = clf_clintClas.fit(X = X_ClintClas_model, y = Y_ClintClas_model.values.tolist())
#%%
###########################################################################
## Intrinsic Clearance Regression
###########################################################################
## Regression
## Extract y data for regression
Y_reg = Y[(Y > 0.9) & (Y <= 50)]
## Transform Y
Y_reg = Y_reg.apply(lambda x: np.log10(x))

## Combine fingerprints and perform feature selection
fingerprints = pd.concat([df_pubchem, df_chemotypes], axis=1).dropna()
fingerprints = fingerprints.loc[Y_reg.index,:].dropna()
#%%
# Select fingerprints from the feature file
retain = [str(val.replace("'", "").replace(" ", "")) for val in clint_features_reg.ix[0,'Fingerprints'].split(',')]
retain[0] = retain[0].replace("[", "")
retain[len(retain)-1] = retain[len(retain)-1].replace("]",'')
fingerprints = fingerprints.loc[:,retain]

descriptors = pd.concat([padel, cdk], axis=1).dropna()
descriptors = descriptors.loc[Y_clas.index,:].dropna()

# Select descriptors from the feature file
retain = [str(val.replace("'", "").replace(" ", "")) for val in clint_features_reg.ix[0,'Padel+CDK'].split(',')]
retain[0] = retain[0].replace("[", "")
retain[len(retain)-1] = retain[len(retain)-1].replace("]",'')
descriptors = descriptors.loc[:,retain]

#%%
## Combine all the descriptors and set the X and Y for training the regression model
data = pd.concat([Y_reg, fingerprints], axis=1).dropna(axis=0, how='any')

data.to_csv(path+'output/clintreg_trainingdata.csv', index_label='CASRN')

X_ClintReg_model = data.ix[:, data.columns != y_var]
Y_ClintReg_model = data[y_var]

## Histogram of final data
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6), dpi = 300)

meanY = np.mean(Y_ClintReg_model)
stdY = np.std(Y_ClintReg_model)

Y_ClintReg_model.hist(alpha = 0.75, color = 'r', grid = False)
plt.annotate('N = %d' %len(Y_ClintReg_model), [-.05,50], size = 20)
plt.annotate('$\mu = %0.2f$' %(meanY), [-.05,45], size = 20)
plt.annotate('$\sigma = %0.2f$' %(stdY), [-.05,40], size = 20)

plt.xlabel('Transformed Clearance (for regression)', size = 24, labelpad = 10)
plt.ylabel('Frequency', size = 24, labelpad = 10)
plt.xticks(fontsize = 24, rotation = 90)
plt.yticks(fontsize = 24)

plt.savefig(path+'/output/%s_TrainingData.png' %y_var, bbox_inches='tight')
plt.show()
#%%
data.to_csv(path+'output/fub_trainingdata.csv', index_label='CASRN')


###########################################################################
## Develop model 

clf_clintReg = RandomForestRegressor(max_features = 'sqrt', n_estimators = 500, random_state = 5)
clf_clintReg = clf_clintReg.fit(X = X_ClintReg_model, y = Y_ClintReg_model)


## Save the models to disk
pickle.dump(clf_clintClas, open(path+'output/clintClas_svc.sav', 'wb'))
pickle.dump(clf_clintReg, open(path+'output/clintReg_rf.sav', 'wb'))

print("all done")