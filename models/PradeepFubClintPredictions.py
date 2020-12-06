# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:51:18 2020

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
import pickle

#%%
###########################################################################
## Set working directory 
###########################################################################
path = 'C:/Users/Administrator/OneDrive/Profile/Desktop/HTTK/'
#path = 'Z:/Projects/HTTK/' 

#%%
###########################################################################
## Load predictive models
###########################################################################
fub_svr = pickle.load(open(path+'output/fub_svr.sav','rb'))
fub_rf = pickle.load(open(path+'output/fub_rf.sav','rb'))
clintClas_svr = pickle.load(open(path+'output/clintClas_svc.sav', 'rb'))
clintReg_rf = pickle.load(open(path+'output/clintReg_rf.sav', 'rb'))

###########################################################################
## Load normalization vector 
###########################################################################
opera_scaler = pickle.load(open(path+'output/opera_scaler.sav', 'rb'))

###########################################################################
## Read Final Features from the 5-fold CV model
###########################################################################
fub_features = pd.read_csv(path+'output/Human.Funbound.plasma_Features.csv')
clint_features_clas = pd.read_csv(path+'output/Clint_Features_Classification.csv')
clint_features_reg = pd.read_csv(path+'output/Clint_Features_Regression.csv')

#%%
###########################################################################
## Read fingerprint and descriptor data for chemicals to be predicted
## Final models need only fingerprints and OPERA predictors
###########################################################################

data=pd.read_csv(path+'predict/set1/HTTK-Chem-Props.txt', sep='\t')
#%%
## Chemotyper FPs: 779 Toxprints 
df_chemotypes = pd.read_csv(path+'predict/set1/HTTK-Chem-Props_toxprints.csv', index_col='M_NAME') #Rename 'M_NAME' to 'CAS' in data file
df_chemotypes = df_chemotypes[~df_chemotypes.index.duplicated(keep='first')]
#%%
## PubChem FPs: 881 bits
df_pubchem = pd.read_csv(path+'predict/set1/HTTK_QSAR_pubchem.csv', index_col='Molecule name')
df_pubchem = df_pubchem[~df_pubchem.index.duplicated(keep='first')]
#%%
# combine fingerprints
fingerprints = pd.concat([df_pubchem, df_chemotypes], axis=1)

#%%
# OPERA 
df_opera = pd.read_csv(path+'predict/set1/OPERA2.5_Pred.csv', index_col='MoleculeID')[['LogP_pred','pKa_a_pred', 'pKa_b_pred']] #In MOE: Right click on mol -> Name -> Extract -> new field 'CAS'
df_opera['pKa_pred']=df_opera[['pKa_a_pred','pKa_b_pred']].min(axis=1)
df_opera = df_opera[~df_opera.index.duplicated(keep='first')]
# Normalize opera properties based on transformation scaler vector from the base models
opera_scaled = opera_scaler.transform(df_opera)
opera = pd.DataFrame(opera_scaled, index = df_opera.index) 
opera.columns = df_opera.columns
opera = opera[['LogP_pred','pKa_pred']]

#%%

##########################################################################
## Fraction unbound predictions
###########################################################################
# Select fingerprints from the feature file
retain = [str(val.replace("'", "").replace(" ", "")) for val in fub_features.ix[0,'Fingerprints'].split(',')]
retain[0] = retain[0].replace("[", "")
retain[len(retain)-1] = retain[len(retain)-1].replace("c]",'c')
fingerprints_fub = fingerprints.loc[:,retain]
# Set X vector for predictions
X_fub = pd.concat([fingerprints_fub, opera], axis=1).dropna()
#%%
###########################################################################
## Clearance Predictions
###########################################################################
# 1. Classification -------------------------------------------------------
# Select fingerprints from the feature file
retain = [str(val.replace("'", "").replace(" ", "")) for val in clint_features_clas.ix[0,'Fingerprints'].split(',')]
retain[0] = retain[0].replace("[", "")
retain[len(retain)-1] = retain[len(retain)-1].replace("]",'')
fingerprints_clintClas = fingerprints.loc[:,retain]
# Set X vector for predictions
X_clintClas = pd.concat([fingerprints_clintClas, opera], axis=1).dropna()

#%%
# 2. Regression -------------------------------------------------------
# Select fingerprints from the feature file
retain = [str(val.replace("'", "").replace(" ", "")) for val in clint_features_reg.ix[0,'Fingerprints'].split(',')]
retain[0] = retain[0].replace("[", "")
retain[len(retain)-1] = retain[len(retain)-1].replace("]",'')
fingerprints_clintReg = fingerprints.loc[:,retain]
# Set X vector for predictions
X_clintReg = pd.concat([fingerprints_clintReg], axis=1)
# Keep only the chemicals for which a chemical bin prediction can be made
X_clintReg = X_clintReg.loc[X_clintClas.index]
#%%
pradeep_FubClintPredictions = pd.DataFrame(index=X_fub.index, columns=['Fub (SVR Prediction)', 'Fub (RF Prediction)', 'Fub (Consensus Prediction)', 'Clint Prediction (Bin)', 'Clint Prediction'])
# Fraction unbound predictions (transformed back to 0-1 range)
pradeep_FubClintPredictions['Fub (SVR Prediction)'] = 1/(1+10**fub_svr.predict(X_fub))
pradeep_FubClintPredictions['Fub (RF Prediction)']  = 1/(1+10**fub_rf.predict(X_fub))
pradeep_FubClintPredictions['Fub (Consensus Prediction)']  = (pradeep_FubClintPredictions['Fub (SVR Prediction)']+pradeep_FubClintPredictions['Fub (RF Prediction)'])/2

#%%
# Clearance predictions
# Classification
pradeep_FubClintPredictions['Clint Prediction (Bin)'] = clintClas_svr.predict(X_clintClas)
# Set the values as low, medium, high
pradeep_FubClintPredictions.loc[pradeep_FubClintPredictions['Clint Prediction (Bin)'] == -3, 'Clint Prediction (Bin)'] = 'Low'
pradeep_FubClintPredictions.loc[pradeep_FubClintPredictions['Clint Prediction (Bin)'] == -2, 'Clint Prediction (Bin)'] = 'Medium'
pradeep_FubClintPredictions.loc[pradeep_FubClintPredictions['Clint Prediction (Bin)'] == -1, 'Clint Prediction (Bin)'] = 'High'
# Regression
pradeep_FubClintPredictions['Clint Prediction'] = 10**(clintReg_rf.predict(X_clintReg))
# Update the predictions for chemicals that are predicted Low or High to default median values
# See model details in the paper. 
pradeep_FubClintPredictions.loc[pradeep_FubClintPredictions['Clint Prediction (Bin)'] == 'Low', 'Clint Prediction'] = 0
pradeep_FubClintPredictions.loc[pradeep_FubClintPredictions['Clint Prediction (Bin)'] == 'High', 'Clint Prediction'] = 102.21000000000001

#%%
pradeep_FubClintPredictions.to_csv(path+'predict/set1/pradeep_FubClintPredictions_HTTK_chem_props.csv', index_label='CASRN')