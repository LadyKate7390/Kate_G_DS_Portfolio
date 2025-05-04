#!/usr/bin/env python
# coding: utf-8

# # Sonic Porosity Log: Predicting DTS and DTC 
# ## Using Wavelet Transforms and ML 
# Sonic Log also called porosity logs are sets of data collected and used by geophyicists to evalute the rock layer properties of well bores.  Sound waves are generated transverse/ compressiona and shearwave in the bore hole or adjecent too and the returning sound is collected.  The delta T ( of time of fight) and wave type can reveal information about the resevoir and rock layers.  This information is vital to maximize the cost and efficiency and production in the oil and gas industry drilling operations.   This project will attempt to show that the shearwave DTS and compressional wave DTC  data which is often missing parts or not collected, yet very vital, can be predicted given other features in the logs. 
# 
# ### Wavelet transform and XGBoost Regression used for predictions

# ## Data is taken from Synthethic Sonic Log Curves Generation Contest 

# #### Description fo the data:
# - CAL - Caliper, unit in Inch,
# - CNC - Neutron, unit in dec
# - GR - Gamma Ray, unit in API
# - HRD - Deep Resisitivity, unit in Ohm per meter,
# - HRM - Medium Resistivity, unit in Ohm per meter,
# - PE - Photo-electric Factor, unit in Barn,
# - ZDEN - Density, unit in Gram per cubit meter,
# - DTC - Compressional Travel-time, unit in nanosecond per foot,
# - DTS - Shear Travel-time, unit in nanosecond per foot,

# In[4]:


pip install xgboost


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

import xgboost as xgb


# In[6]:


# Load in the train and test data sets from GitHub 
urltr = 'https://raw.githubusercontent.com/LadyKate7390/Kate_G_DS_Portfolio/refs/heads/main/Sonic%20Wave%20Velocity/train.csv'
urltest = 'https://raw.githubusercontent.com/LadyKate7390/Kate_G_DS_Portfolio/refs/heads/main/Sonic%20Wave%20Velocity/test.csv'
url = 'https://raw.githubusercontent.com/LadyKate7390/Kate_G_DS_Portfolio/refs/heads/main/Sonic%20Wave%20Velocity/real_test_result.csv'
tr_df = pd.read_csv(urltr)
ts_df= pd.read_csv(urltest)


# In[7]:


print(tr_df.shape,ts_df.shape)


# In[8]:


tr_df.describe()


# In[ ]:





# In[9]:


# make a copy of train set as we clean up some data
dft = tr_df.copy()


# In[10]:


# replace negative -999 values and deal with missing data (Na) 
dft.replace(['-999', -999], np.nan, inplace=True)


# In[11]:


nan_counts = dft.isnull().sum()
print(nan_counts)


# In[12]:


histdf = dft.hist(bins=100,figsize=(15,10))


# In[13]:


dft.plot(subplots=True,figsize=(20,10))


# In[14]:


#Making negative values Nan of ZDEN, GR, CNC, PE. We don't want to just drop them or make them '0' for now
col = ['ZDEN', 'GR', 'CNC', 'PE']
dft[col] = dft[col].mask(dft[col] < 0)



# In[15]:


plt.figure(figsize=(13,10))
plt.subplot(4,2,1)
sns.boxplot(dft['CAL'])

plt.subplot(4,2,2)
sns.boxplot(dft['CNC'])

plt.subplot(4,2,3)
sns.boxplot(dft['GR'])

plt.subplot(4,2,4)
sns.boxplot(dft['HRD'])

plt.subplot(4,2,5)
sns.boxplot(dft['HRM'])

plt.subplot(4,2,6)
sns.boxplot(dft['PE'])

plt.subplot(4,2,7)
sns.boxplot(dft['ZDEN'])

plt.tight_layout()
plt.show()


# In[16]:


# transforming a few variables with log function and replotting 
dft['HRM'] = dft['HRM'].apply(lambda x:np.log(x))
dft['HRD'] = dft['HRD'].apply(lambda x:np.log(x))


# In[17]:


plt.figure(figsize=(13,10))
plt.subplot(4,2,1)
sns.boxplot(dft['CAL'])

plt.subplot(4,2,2)
sns.boxplot(dft['CNC'])

plt.subplot(4,2,3)
sns.boxplot(dft['GR'])

plt.subplot(4,2,4)
sns.boxplot(dft['HRD'])

plt.subplot(4,2,5)
sns.boxplot(dft['HRM'])

plt.subplot(4,2,6)
sns.boxplot(dft['PE'])

plt.subplot(4,2,7)
sns.boxplot(dft['ZDEN'])

plt.tight_layout()
plt.show()


# In[18]:


#Looking for correlations 
dft.corr()


# In[19]:


ts_df.corr()


# In[20]:


# Correlation Matrix 
corr_matrix = dft.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# The intent was to use ppscore but issues with instalation with the library has led to using an alrenative MIC as the pearson method above is best suited for linear relationships which this data is not. 
# 

# In[22]:


corr_matrix = dft.corr(method='kendall')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Running two differnt correlations using two different scoring types one for more linear relationships 
# the other not.  Some slight differences in scoring is note. The Kendall method gave some stronger correlations between some features suggesting that there are strong non-linear relationships exist. 

# In[24]:


# Going to try and run ppscore now that older version of python installed. 
import ppscore as pps
matrix_df = pps.matrix(dft).pivot(columns='x', index='y',  values='ppscore')
sns.heatmap(matrix_df, vmin=0, vmax=1,  linewidths=1.0,annot=True)


# Getting to run ppscore along with the older ways of doing correlation matrix it is clear that even when using differnt methods ppscore is far better at non-linear complext relationships within a dataframe.

# # Building the Models 

# In[27]:


# Create separate datasets for DTC and DTS
df_dtc = dft.dropna(subset=['DTC'])
df_dts = dft.dropna(subset=['DTS'])


# In[28]:


df_dtc= df_dtc.dropna()
df_dts= df_dts.dropna()


# In[29]:


#X
df_dtc_x = df_dtc.drop(columns=['DTC','DTS'])
df_dts_x = df_dts.drop(columns=['DTC','DTS'])


# In[30]:


#Y
y_dtc = df_dtc['DTC']
y_dts = df_dts['DTS']


# In[31]:


X_train_dtc, X_test_dtc, y_train_dtc, y_test_dtc = train_test_split(df_dtc_x,y_dtc, test_size=0.30, random_state=42, shuffle = True)


# In[32]:


X_train_dts, X_test_dts, y_train_dts, y_test_dts = train_test_split(df_dts_x,y_dts, test_size=0.30, random_state=42, shuffle = True)


# #  Hyper parameters used in X G Boost 
# ## Training dataframe for DTS,DTC

# # DTC

# In[35]:


xgb_model_dtc = xgb.XGBRegressor(random_state=42, max_depth=2,learning_rate=0.18, n_estimators=145, min_child_weight = 6, gamma = 0.3)


# In[36]:


xgb_model_dtc.fit(X_train_dtc, y_train_dtc)
y_pred_test_dtc = xgb_model_dtc.predict(X_test_dtc)
y_pred_train_dtc = xgb_model_dtc.predict(X_train_dtc)
print("RMSE_train:     " + str(np.sqrt(mean_squared_error(y_train_dtc,y_pred_train_dtc))))
print("RMSE_test:     " + str(np.sqrt(mean_squared_error(y_test_dtc,y_pred_test_dtc))))
print("R2_train:     " + str(r2_score(y_train_dtc,y_pred_train_dtc)))
print("R2_test:     " + str(r2_score(y_test_dtc,y_pred_test_dtc)))


# In[37]:


xgb.plot_importance(xgb_model_dtc)


# # DTS

# In[39]:


xgb_model_dts = xgb.XGBRegressor(random_state=42, max_depth=7,learning_rate=0.19, n_estimators=135, min_child_weight = 6, gamma = 0.7)


# In[40]:


xgb_model_dts.fit(X_train_dts, y_train_dts)
y_pred_test_dts = xgb_model_dts.predict(X_test_dts)
y_pred_train_dts = xgb_model_dts.predict(X_train_dts)
print("RMSE_train:     " + str(np.sqrt(mean_squared_error(y_train_dts,y_pred_train_dts))))
print("RMSE_test:     " + str(np.sqrt(mean_squared_error(y_test_dts,y_pred_test_dts))))
print("R2_train:     " + str(r2_score(y_train_dts,y_pred_train_dts)))
print("R2_test:     " + str(r2_score(y_test_dts,y_pred_test_dts)))


# In[41]:


xgb.plot_importance(xgb_model_dts)


# # Test Data 
#      Running predictions using the test data set

# In[43]:


# Preparing the test dataframe 
# Replace value -999 ( missing value indicators ) as NA
ts_df.replace(['-999', -999], np.nan, inplace=True)
# Nullify the Negative Values
col = ['ZDEN', 'GR', 'CNC', 'PE']
ts_df[col] = ts_df [col].mask(dft[col] < 0)

# Nullify the Outliers
ts_df['CNC'][ts_df['CNC']>0.7] = np.nan
ts_df['GR'][(ts_df['GR']>250)] = np.nan
ts_df['HRD'][ts_df['HRD']>200] = np.nan
ts_df['HRM'][ts_df['HRM']>200] = np.nan

        
# Log Transformation
ts_df['HRD'] = np.log(ts_df['HRD'])
ts_df['HRM'] = np.log(ts_df['HRM'])


# In[44]:


ts_df.isnull().sum()


# In[45]:


ts_df['GR']= ts_df['GR'].fillna(ts_df['GR'].mean())
ts_df['HRD']= ts_df['HRD'].fillna(ts_df['HRD'].mean())
ts_df['HRM']= ts_df['HRM'].fillna(ts_df['HRM'].mean())


# In[46]:


ts_df.isnull().sum()


# In[47]:


ts_df.shape


# In[48]:


test_x= ts_df.copy()


# In[49]:


ts_df['DTC'] = xgb_model_dtc.predict(test_x)
ts_df['DTS'] = xgb_model_dts.predict(test_x)


# In[50]:


ts_df


# In[51]:


# Reading in orginal dataset ( 
dfr = pd.read_csv(url)
dfr.shape


# In[52]:


pred_df= ts_df[['DTC','DTS']]
pred_df.shape


# In[53]:


preds= np.array(pred_df)
reals= np.real(dfr)


# In[54]:


def result_plot(y_predict, y_real):
    # check the accuracy of predicted data and plot the result
    print('Combined r2 score is:', '{:.5f}'.format((r2_score(y_real, y_predict))))
    dtc_real = y_real[:, 0]
    dtc_pred = y_predict[:, 0]
    dts_real = y_real[:, 1]
    dts_pred = y_predict[:, 1]
    print('DTC:', '{:.5f}'.format((r2_score(dtc_real, dtc_pred))))
    print('DTS:', '{:.5f}'.format((r2_score(dts_real, dts_pred))))
    plt.subplots(nrows=2, ncols=2, figsize=(16,10))
    plt.subplot(2, 2, 1)
    plt.plot(y_real[:, 0])
    plt.plot(y_predict[:, 0])
    plt.legend(['True', 'Predicted'])
    plt.xlabel('Sample')
    plt.ylabel('DTC')
    plt.title('DTC Prediction Comparison')

    plt.subplot(2, 2, 2)
    plt.plot(y_real[:, 1])
    plt.plot(y_predict[:, 1])
    plt.legend(['True', 'Predicted'])
    plt.xlabel('Sample')
    plt.ylabel('DTS')
    plt.title('DTS Prediction Comparison')
    
    plt.subplot(2, 2, 3)
    plt.scatter(y_real[:, 0], y_predict[:, 0])
    plt.xlabel('Real Value')
    plt.ylabel('Predicted Value')
    plt.title('DTC Prediction Comparison')
    
    plt.subplot(2, 2, 4)
    plt.scatter(y_real[:, 1], y_predict[:, 1])
    plt.xlabel('Real Value')
    plt.ylabel('Predicted Value')
    plt.title('DTS Prediction Comparison')

    plt.show()


# In[55]:


result_plot(preds,reals)


# In[56]:


# Plot results:
plt.figure(figsize=(15,5))
i = 0
plt.subplot(1,2,i+1)
plt.plot(preds[:,i], reals[:,i], '.', label = 'r^2 = %.3f' % ((r2_score(reals[:,i], preds[:,i]))))
plt.plot([reals[:,i].min(),reals[:,i].max()],[reals[:,i].min(),reals[:,i].max()], 'r', label = '1:1 line')
plt.title('#1 DTC: y_true vs. y_ estimate'); plt.xlabel('Estimate'); plt.ylabel('True')
plt.legend()
i += 1
plt.subplot(1,2,i+1)
plt.plot(preds[:,i], reals[:,i], '.', label = 'r^2 = %.3f' % ((r2_score(reals[:,i], preds[:,i]))))
plt.plot([reals[:,i].min(),reals[:,i].max()],[reals[:,i].min(),reals[:,i].max()], 'r', label = '1:1 line')
plt.title('#2 DTS: y_true vs. y_ estimate'); plt.xlabel('Estimate'); plt.ylabel('True')
plt.legend()

MSE_0 = mean_squared_error(reals[:,0], preds[:,0]);
RMSE_0 = np.sqrt(mean_squared_error(reals[:,0], preds[:,0]));
MSE_1 = mean_squared_error(reals[:,1], preds[:,1]);
RMSE_1 = np.sqrt(mean_squared_error(reals[:,1], preds[:,1]));
print('RMSE of test data (#1 DTC): %.2f' %(RMSE_0))
print('RMSE of test data (#2 DTS): %.2f' %(RMSE_1))
print('Overall RMSE = %.2f' %np.sqrt((MSE_0+MSE_1)/2))


# # Using Pycaret

# In[58]:


from pycaret.regression import *


# In[59]:


df_dtspycaret= df_dts.drop(['DTC'],axis=1)


# In[60]:


model= setup(data= df_dtspycaret, target= 'DTS',normalize= True,remove_outliers=True,profile=True)


# In[61]:


xgb= create_model('xgboost')


# In[62]:


plot_model(xgb)


# # For DTC

# In[64]:


df_dtcpycaret= df_dtc.drop(['DTS'],axis=1)


# In[65]:


model= setup(data= df_dtspycaret, target= 'DTS',normalize= True,remove_outliers=True,profile=True)


# In[66]:


xgb= create_model('xgboost')


# In[67]:


plot_model(xgb)


# # Wavelt Transform
#     So far the data has been approached from typical approaces.  Doing this will decompose the signal into frequency components which 
#     will allow for more detailed analysis of the sonic logs.  This decompostion seperates out the temporal and sparial features which in turn can lead to imporved feature extractions. 

# In[69]:


import shap
# load JS visualization code to notebook
shap.initjs()
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(xgb_model_dts)


# In[70]:


shap_values = explainer.shap_values(test_x)


# In[71]:


# summarize the effects of all the features
shap.summary_plot(shap_values, test_x)


# In[72]:


#This function is defined for descrete wavelet transform and calculating the cA ie the approximate coefficient
def make_dwt_vars_cA(wells_df,logs,levels,wavelet):

    wave= pywt.Wavelet(wavelet)
    
    grouped = wells_df
    new_df = pd.DataFrame()
    for key in grouped.keys():
    
        depth = grouped['Depth']
        temp_df = pd.DataFrame()
        temp_df['Depth'] = depth
        for log in logs:
      
            temp_data = grouped[log]
              
            for i in levels:
                
                    cA_cD = pywt.wavedec(temp_data,wave,level=i,mode='symmetric')
                    cA = cA_cD[0]
                    new_depth = np.linspace(min(depth),max(depth),len(cA))
                    fA = interp1d(new_depth,cA,kind='nearest')
                    temp_df[log + '_cA_level_' + str(i)] = fA(depth)
    
        new_df = new_df.append(temp_df)
        
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df


# In[73]:


#This is a user defined function for discrte wavelet transform cD, cD is also called the detailed  coefficient
## There is no depth feature so this creates some
def make_dwt_vars_cD(wells_df,logs,levels,wavelet):

    wave= pywt.Wavelet(wavelet)
    
    grouped = wells_df
    new_df = pd.DataFrame()
    for key in grouped.keys():
    
        depth = grouped['Depth']
        temp_df = pd.DataFrame()
        temp_df['Depth'] = depth
        for log in logs:
      
            temp_data = grouped[log]

            cA_4, cD_4, cD_3, cD_2, cD_1 = pywt.wavedec(temp_data,wave,level=4,mode='symmetric')
            dict_cD_levels = {1:cD_1, 2:cD_2, 3:cD_3, 4:cD_4}
                
            for i in levels:
                new_depth = np.linspace(min(depth),max(depth),len(dict_cD_levels[i]))
                fA = interp1d(new_depth,dict_cD_levels[i],kind='nearest')
                temp_df[log + '_cD_level_' + str(i)] = fA(depth)
    
        new_df = new_df.append(temp_df)
        
    new_df = new_df.sort_index()
    new_df = new_df.drop(['Depth'],axis=1)
    return new_df


# Getting the "TRAIN" data ready 

# In[75]:


df_wavelet= dft.copy()


# In[76]:


df_wavelet= df_wavelet.dropna()


# In[77]:


df_wavelet.shape


# In[78]:


depth_train= np.linspace(500,4000,len(df_wavelet))


# In[79]:


df_wavelet['Depth']= depth_train


# # dB4 Wavelet Transformation 

# In[81]:


import pywt
from scipy.interpolate import interp1d


# In[82]:


#cD From wavelet db4
dwt_db4_cD_df = make_dwt_vars_cD(wells_df=df_wavelet, logs=['CNC'],levels=[1, 2, 3, 4], wavelet='db4')

# cA From wavelet db4
dwt_db4_cA_df = make_dwt_vars_cA(wells_df=df_wavelet, logs=['CNC'],levels=[1, 2, 3, 4], wavelet='db4')


# In[83]:


list_df_var = [dwt_db4_cD_df, dwt_db4_cA_df]
combined_df = df_wavelet
for var_df in list_df_var:
    temp_df = var_df
    combined_df = pd.concat([combined_df,temp_df],axis=1)
combined_df.replace(to_replace=np.nan, value='-1', inplace=True)
print (combined_df.shape)
training_data=combined_df


# In[84]:


df_zone1= training_data[training_data['Depth']<1000]


# In[85]:


f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 16))
ax[0].plot(df_zone1.CNC, df_zone1.Depth, '-', color='black')
ax[1].plot(df_zone1.CNC_cD_level_1, df_zone1.Depth, '-g')
ax[2].plot(df_zone1.CNC_cD_level_2, df_zone1.Depth, '-')
ax[3].plot(df_zone1.CNC_cD_level_3, df_zone1.Depth, '-', color='0.5')
ax[4].plot(df_zone1.CNC_cD_level_4, df_zone1.Depth, '-', color='r')

ax[0].set_xlabel("CNC Log")
ax[0].set_xlim(df_zone1.CNC.min(),df_zone1.CNC.max())
ax[1].set_xlabel("CNC_cD_level_1")
ax[1].set_xlim(df_zone1.CNC_cD_level_1.min(),df_zone1.CNC_cD_level_1.max())
ax[2].set_xlabel("CNC_cD_level_2")
ax[2].set_xlim(df_zone1.CNC_cD_level_2.min(),df_zone1.CNC_cD_level_2.max())
ax[3].set_xlabel("CNC_cD_level_3")
ax[3].set_xlim(df_zone1.CNC_cD_level_3.min(),df_zone1.CNC_cD_level_3.max())
ax[4].set_xlabel("CNC_cD_level_4")
ax[4].set_xlim(df_zone1.CNC_cD_level_4.min(),df_zone1.CNC_cD_level_4.max())

ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
ax[4].set_yticklabels([]); 

f.suptitle('CMC log and corresponding cD cofficients found via db4 for Well: ', fontsize=14,y=0.94)


# In[86]:


# Create separate datasets for DTC and DTS
df_dtc_wave = training_data.dropna(subset=['DTC'])
df_dts_wave = training_data.dropna(subset=['DTS'])


# In[87]:


df_dtc_x_wave = df_dtc_wave.drop(columns=['DTC','DTS'])
df_dts_x_wave = df_dts_wave.drop(columns=['DTC','DTS'])


# In[88]:


y_dtc_wave = df_dtc_wave['DTC']
y_dts_wave = df_dts_wave['DTS']


# In[89]:


X_train_dtc_wave, X_test_dtc_wave, y_train_dtc_wave, y_test_dtc_wave = train_test_split(df_dtc_x_wave,y_dtc_wave, test_size=0.30, random_state=42, shuffle = True)


# In[90]:


X_train_dts_wave, X_test_dts_wave, y_train_dts_wave, y_test_dts_wave = train_test_split(df_dts_x_wave,y_dts_wave, test_size=0.30, random_state=42, shuffle = True)


# # Now for DTC

# In[92]:


from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

xgb_model_dtc_wave = xgb.XGBRegressor(random_state=42, max_depth=2,learning_rate=0.18, n_estimators=145, min_child_weight = 6, gamma = 0.3)


# In[93]:


xgb_model_dtc_wave.fit(X_train_dtc_wave, y_train_dtc_wave)
y_pred_test_dtc_wave = xgb_model_dtc_wave.predict(X_test_dtc_wave)
y_pred_train_dtc_wave = xgb_model_dtc_wave.predict(X_train_dtc_wave)
print("RMSE_train:     " + str(np.sqrt(mean_squared_error(y_train_dtc_wave,y_pred_train_dtc_wave))))
print("RMSE_test:     " + str(np.sqrt(mean_squared_error(y_test_dtc_wave,y_pred_test_dtc_wave))))
print("R2_train:     " + str(r2_score(y_train_dtc_wave,y_pred_train_dtc_wave)))
print("R2_test:     " + str(r2_score(y_test_dtc_wave,y_pred_test_dtc_wave)))


# # DTS

# In[95]:


xgb_model_dts_wave = xgb.XGBRegressor(random_state=42, max_depth=7,learning_rate=0.19, n_estimators=135, min_child_weight = 6, gamma = 0.7)


# In[96]:


xgb_model_dts_wave.fit(X_train_dts_wave, y_train_dts_wave)
y_pred_test_dts_wave = xgb_model_dts_wave.predict(X_test_dts_wave)
y_pred_train_dts_wave = xgb_model_dts_wave.predict(X_train_dts_wave)
print("RMSE_train:     " + str(np.sqrt(mean_squared_error(y_train_dts_wave,y_pred_train_dts_wave))))
print("RMSE_test:     " + str(np.sqrt(mean_squared_error(y_test_dts_wave,y_pred_test_dts_wave))))
print("R2_train:     " + str(r2_score(y_train_dts_wave,y_pred_train_dts_wave)))
print("R2_test:     " + str(r2_score(y_test_dts_wave,y_pred_test_dts_wave)))


# ## Predictions for test data 

# In[98]:


merge= [test_x,dfr]
wavelet_test_df= pd.concat(merge,axis=1)
print(wavelet_test_df.shape)
wavelet_test_df


# In[99]:


depth_test= np.linspace(500,4000,len(wavelet_test_df))
wavelet_test_df['Depth']= depth_test
wavelet_test_df.isnull().sum()


# In[100]:


#cD From wavelet db4
dwt_db4_cD_df = make_dwt_vars_cD(wells_df=wavelet_test_df, logs=['CNC'],levels=[1, 2, 3, 4], wavelet='db4')

# cA From wavelet db4
dwt_db4_cA_df = make_dwt_vars_cA(wells_df=wavelet_test_df, logs=['CNC'],levels=[1, 2, 3, 4], wavelet='db4')


# In[101]:


list_df_var = [dwt_db4_cD_df, dwt_db4_cA_df]
combined_df = wavelet_test_df
for var_df in list_df_var:
    temp_df = var_df
    combined_df = pd.concat([combined_df,temp_df],axis=1)
combined_df.replace(to_replace=np.nan, value='-1', inplace=True)
print (combined_df.shape)
testing_data=combined_df


# In[102]:


testing_data.columns
final_test_df= testing_data.drop(['DTC    ','DTS  '],axis=1)
final_test_df_dtc= final_test_df.copy()
final_test_df_dts= final_test_df.copy()


# ### Predictions

# In[104]:


X_train_dtc_wave.columns
final_test_df_dtc.columns
final_test_df_dts.columns
final_test_df['DTC']= xgb_model_dtc_wave.predict(final_test_df_dtc)
final_test_df['DTS']= xgb_model_dts_wave.predict(final_test_df_dts)
final_test_df


# In[105]:


testing_data


# In[106]:


testing_data.columns


# In[107]:


pred_wave_df= final_test_df[['DTC','DTS']]
real_wave_df= testing_data[['DTC    ','DTS  ']]
preds_wave= np.array(pred_wave_df)
reals_wave= np.array(real_wave_df)
merge1= [testing_data['DTS  '],final_test_df['DTS']]
dts= pd.concat(merge1,axis=1)
dts['res']= dts['DTS  ']-dts['DTS']
plt.plot(dts['res'])


# #### Trying wavelet on another feature: Caliper log for DTS prediction

# In[212]:


df_wavelet_cal= dft.copy()
df_wavelet_cal= df_wavelet_cal.dropna()
depth_train_cal= np.linspace(500,4000,len(df_wavelet_cal))
df_wavelet_cal['Depth']= depth_train_cal
#cD From wavelet db4
dwt_db4_cD_df_cal = make_dwt_vars_cD(wells_df=df_wavelet_cal, logs=['CAL'],levels=[1, 2, 3, 4], wavelet='db4')

# cA From wavelet db4
dwt_db4_cA_df_cal = make_dwt_vars_cA(wells_df=df_wavelet_cal, logs=['CAL'],levels=[1, 2, 3, 4], wavelet='db4')
list_df_var_cal = [dwt_db4_cD_df_cal, dwt_db4_cA_df_cal]
combined_df_cal = df_wavelet_cal
for var_df_cal in list_df_var_cal:
    temp_df_cal = var_df_cal
    combined_df_cal = pd.concat([combined_df_cal,temp_df_cal],axis=1)
combined_df_cal.replace(to_replace=np.nan, value='-1', inplace=True)
print (combined_df_cal.shape)
training_data_cal=combined_df_cal

df_dts_x_wave_cal = training_data_cal.drop(columns=['DTC','DTS'])
y_dts_wave_cal = training_data_cal['DTS']

X_train_dts_wave_cal, X_test_dts_wave_cal, y_train_dts_wave_cal, y_test_dts_wave_cal = train_test_split(df_dts_x_wave_cal,y_dts_wave_cal, test_size=0.30, random_state=42, shuffle = True)

xgb_model_dts_wave_cal = xgb.XGBRegressor(random_state=42, max_depth=7,learning_rate=0.19, n_estimators=135, min_child_weight = 6, gamma = 0.7)
xgb_model_dts_wave_cal.fit(X_train_dts_wave_cal, y_train_dts_wave_cal)
y_pred_test_dts_wave_cal = xgb_model_dts_wave_cal.predict(X_test_dts_wave_cal)
y_pred_train_dts_wave_cal = xgb_model_dts_wave_cal.predict(X_train_dts_wave_cal)
print("RMSE_train:     " + str(np.sqrt(mean_squared_error(y_train_dts_wave_cal,y_pred_train_dts_wave_cal))))
print("RMSE_test:     " + str(np.sqrt(mean_squared_error(y_test_dts_wave_cal,y_pred_test_dts_wave_cal))))
print("R2_train:     " + str(r2_score(y_train_dts_wave_cal,y_pred_train_dts_wave_cal)))
print("R2_test:     " + str(r2_score(y_test_dts_wave_cal,y_pred_test_dts_wave_cal)))




# In[214]:


# Finding instances of Variable tr_df
wavelet_test_df= wavelet_test_df[2000:]
wavelet_test_df


# In[216]:


#cD From wavelet db4
dwt_db4_cD_df = make_dwt_vars_cD(wells_df=wavelet_test_df, logs=['CAL'],levels=[1, 2, 3, 4], wavelet='db4')

# cA From wavelet db4
dwt_db4_cA_df = make_dwt_vars_cA(wells_df=wavelet_test_df, logs=['CAL'],levels=[1, 2, 3, 4], wavelet='db4')


list_df_var = [dwt_db4_cD_df, dwt_db4_cA_df]
combined_df = wavelet_test_df
for var_df in list_df_var:
    temp_df = var_df
    combined_df = pd.concat([combined_df,temp_df],axis=1)
combined_df.replace(to_replace=np.nan, value='-1', inplace=True)
print (combined_df.shape)
testing_data=combined_df



# In[218]:


testing_data_cal=combined_df

final_test_df_cal= testing_data_cal.drop(['DTC    ','DTS  '],axis=1)
final_test_df_dts_cal= final_test_df_cal.copy()

final_test_df_dts_cal['DTS']= xgb_model_dts_wave_cal.predict(final_test_df_dts_cal)

#print(r2_score(testing_data_cal['DTS  '],final_test_df_cal['DTS']))


# In[220]:


final_test_df_dts_cal['DTS']


# In[222]:


final_test_df_dts_cal['DTS']


# In[224]:


print(np.sqrt(r2_score(testing_data_cal['DTS  '],final_test_df_dts_cal['DTS'])))


# In[226]:


merge1= [testing_data_cal['DTS  '],final_test_df_dts_cal['DTS']]
dts1= pd.concat(merge1,axis=1)


# In[228]:


dts1['res']= dts1['DTS  ']-dts1['DTS']


# In[230]:


plt.plot(dts1['res'])


# In[232]:


# check the accuracy of predicted data and plot the result
#print('Combined r2 score is:', '{:.5f}'.format((r2_score(y_real, y_predict))))
dtc_real = reals_wave[:, 0]
dtc_pred = preds_wave[:, 0]
dts_real = testing_data_cal['DTS  ']
dts_pred = final_test_df_dts_cal['DTS']
print('DTC:', '{:.5f}'.format(np.sqrt(r2_score(dtc_real, dtc_pred))))
print('DTS:', '{:.5f}'.format(np.sqrt(r2_score(dts_real, dts_pred))))
plt.subplots(nrows=2, ncols=2, figsize=(16,10))
plt.subplot(2, 2, 1)
plt.plot(reals_wave[:, 0])
plt.plot(preds_wave[:, 0])
plt.legend(['True', 'Predicted'])
plt.xlabel('Sample')
plt.ylabel('DTC')
plt.title('DTC Prediction Comparison')
    
plt.subplot(2, 2, 2)
plt.plot(testing_data_cal['DTS  '])
plt.plot(final_test_df_dts_cal['DTS'])
plt.legend(['True', 'Predicted'])
plt.xlabel('Sample')
plt.ylabel('DTS')
plt.title('DTS Prediction Comparison')
    
plt.subplot(2, 2, 3)
plt.scatter(reals_wave[:, 0], preds_wave[:, 0])
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('DTC Prediction Comparison')
    
plt.subplot(2, 2, 4)
plt.scatter(testing_data_cal['DTS  '], final_test_df_dts_cal['DTS'])
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.title('DTS Prediction Comparison')

plt.show()


# In[234]:


# Plot results:
plt.figure(figsize=(15,5))
i = 0
plt.subplot(1,2,i+1)
plt.plot(preds_wave[:,i], reals_wave[:,i], '.', label = 'r^2 = %.3f' % (np.sqrt(r2_score(reals_wave[:,i], preds_wave[:,i]))))
plt.plot([reals_wave[:,i].min(),reals_wave[:,i].max()],[reals_wave[:,i].min(),reals_wave[:,i].max()], 'r', label = '1:1 line')
plt.title('#1 DTC: y_true vs. y_ estimate'); plt.xlabel('Estimate'); plt.ylabel('True')
plt.legend()
i += 1
plt.subplot(1,2,i+1)
plt.plot(final_test_df_dts_cal['DTS'], testing_data_cal['DTS  '], '.', label = 'r^2 = %.3f' % np.sqrt((r2_score(testing_data_cal['DTS  '], final_test_df_dts_cal['DTS']))))
plt.plot([testing_data_cal['DTS  '].min(),testing_data_cal['DTS  '].max()],[testing_data_cal['DTS  '].min(),testing_data_cal['DTS  '].max()], 'r', label = '1:1 line')
plt.title('#2 DTS: y_true vs. y_ estimate'); plt.xlabel('Estimate'); plt.ylabel('True')
plt.legend()

MSE_0 = mean_squared_error(reals_wave[:,0], preds_wave[:,0]);
RMSE_0 = np.sqrt(mean_squared_error(reals_wave[:,0], preds_wave[:,0]));
MSE_1 = mean_squared_error(testing_data_cal['DTS  '], final_test_df_dts_cal['DTS']);
RMSE_1 = np.sqrt(mean_squared_error(testing_data_cal['DTS  '], final_test_df_dts_cal['DTS']));
print('RMSE of test data (#1 DTC): %.2f' %(RMSE_0))
print('RMSE of test data (#2 DTS): %.2f' %(RMSE_1))
print('Overall RMSE = %.2f' %np.sqrt((MSE_0+MSE_1)/2))


# In[236]:


meandtc= np.sqrt(r2_score(dtc_real, dtc_pred))
meandts= np.sqrt(r2_score(dts_real, dts_pred))             
test_list= [meandtc,meandts]
sum = 0
for ele in test_list:
  sum += ele
res = sum / len(test_list)


# In[238]:


print("Combined r2 after wavelet transformation is :",res)


# In[ ]:




