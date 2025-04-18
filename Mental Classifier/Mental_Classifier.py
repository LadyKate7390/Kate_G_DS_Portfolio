#!/usr/bin/env python
# coding: utf-8

# # Mental Disorder Classifier 

# # The Data:  
# ## This dataset if from Kaggle and downloaded and stored in my GitHub Project folder
#     It is combosed of 120 Pyschology Patients with 17 essential Symptoms and can be used for the following conditions: 
#     1)Mania Biploar Disorder 
#     2) Depressive Bipolar Disorder 
#     3) Major Depressive Disorder 
#     4) An individual who does not have any of the above but might have another disorder. 
# The symptoms in the dataset are the levels of: Sadness, Exhaustness, Euphoric, Sleep Disorder, Mood Swings, Suicidal Ideations, Anorexia, Anxiet, Try-explaining, Nervouse Breakdown, Ingnore/Move-on, Admitting Mistakes, Overthinking, Agressive Response, Optimism, Sexual activity and Concentration.  
# 
# This project will flow as follows: 
# EDA
#     Feature Engineering 
#     Discritizing variables 
#         Modeling :Random Forest, Support Vector Machine (SVM), Naive Bayes, k-Nearest Neighbors (KNN), Decision Tree,          Gradient Boosting, and Logistic Regression
#             Cross Validatation compare performance. 

# In[5]:


#Import need Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[6]:


#Machine Leraning Libraries 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[7]:


# Importing the dataset as CSV file from GitHub 
url = 'https://raw.githubusercontent.com/LadyKate7390/Kate_G_DS_Portfolio/refs/heads/main/Mental%20Classifier/Dataset-Mental-Disorders.csv'
df= pd.read_csv(url)


# In[8]:


df.head()


# In[21]:


#makeing a copy to use while maintaning orginal downlaod to be called later if needed
df_m =df
df_m.head()


# In[22]:


#Dropping Panient Number as it is not needed 
df_m=df_m.drop('Patient Number' , axis=1)


# In[23]:


print(df_m.dtypes)# to see what variable types we have 


# # All of the variables are object
#     Next the unique types will be looked at and then a scheme can be devloped to discritize these into intigers for modeling later. 

# In[25]:


# Function to iterate throught he dataset and find and print each variable and it unique values 
for column in df_m.columns:
    unique_count =df_m[column].unique()
    unique_value = df_m[column].unique()
    print(f"The number of unique values in {column} : {unique_count}")
    print(f"Unique values in {column}:")
    for  value in unique_value:
          print(value)
    print('\n') 



# #Create Numeric Values for Object variables and addressing missing data. 

# In[27]:


df_m['Suicidal thoughts'] = df_m['Suicidal thoughts'].replace('YES ', "YES") # addressing this binary set as it had three values because of formating error
df_m['Suicidal thoughts'].value_counts()


# In[28]:


# Missing Data 
mis_data= df_m.isnull().sum()
mis_data


# In[29]:


# Type casting Expert Diagnose from String to Intiger which each diagnosis have distinct value 
mapp_diga = {'Normal': 0, 'Bipolar Type-1': 1, 'Bipolar Type-2': 2, 'Depression': 3} #dictionary for mapping 
df_m['Expert Diagnose'] = df_m['Expert Diagnose'].map(mapp_diga).astype(int)
df_m.head()


# In[31]:


#Casting binary yes and no variable to 1 and 0 respectively 
Y_N_Col = ['Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect', 'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down', 'Admit Mistakes', 'Overthinking']
for column in Y_N_Col:
    df_m[column] = df_m[column].map({'YES': 1, 'NO': 0}).astype(int)

df_m.head()


# In[32]:


# Correct Spelling 
df_m = df_m.rename(columns={'Optimisim': 'Optimism'})
df_m = df_m.rename(columns={'Anorxia': 'Anorexia'})


# In[34]:


#Dropping the " from 10" in the following columns
from_col = ['Sexual Activity', 'Concentration', 'Optimism']
for column in from_col:
   df_m[column] = df_m[column].astype(str).str.extract('(\d)')
   df_m[column] = pd.to_numeric(df_m[column])

df_m.head()


# Only one mor type of data to cast to integer from string the frequency types of "sometimes' 'Seldom', 'Often' etc that is found in three of the variables. After which EDA can be beformed. 

# In[35]:


freqcol = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']
for column in freqcol:
    df_m[column] = df_m[column].map({'Seldom': 0, 'Sometimes': 1, 'Usually': 2, 'Most-Often': 3}).astype(int)

df_m.head()


# In[49]:


#Pandas Stats 
df_m.describe()


# In[50]:


grid = sns.FacetGrid(df_m, col='Suicidal thoughts', row='Expert Diagnose', height=2.2, aspect=2.2)
grid.map(plt.hist, 'Sadness')
grid.add_legend()


# In[51]:


import seaborn as sns 
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

correlation_matrix = df_m.corr()

# create a heatmap
plt.figure(figsize=(14, 14))
heatmap = sns.heatmap(correlation_matrix, annot = True, fmt = ".2f", cmap = 'coolwarm')

plt.title('Correlation Heatmap')
plt.show()


# #Modeling 

# In[56]:


#Column headers
column_headers = list(df_m.columns)
print(column_headers)


# In[55]:


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import time

model_performance = pd.DataFrame(columns=['Accuracy', 'Precision',
                                          'Recall', 'F1-Score', 'Training time',
                                          'Prediction time'])
def log_scores(model_name, y_test, y_predictions):
    accuracy = accuracy_score(y_test, y_predictions)
    precision = precision_score(y_test, y_predictions, average='weighted')
    recall = recall_score(y_test, y_predictions, average='weighted')
    precision = precision_score(y_test, y_predictions, average='weighted')
    f1 = f1_score(y_test, y_predictions, average='weighted')

    # save the scores in model_performance dataframe
    model_performance.loc[model_name] = [accuracy, precision, recall, f1,
                                       end_train-start, end_predict-end_train]


# In[62]:


#Creat a 70% train and 30% test
from sklearn.model_selection import train_test_split

X = df_m.drop(['Expert Diagnose'], axis=1)
y = df_m['Expert Diagnose']

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3, 
                                                    random_state = 0,
                                                    stratify = y)


# In[63]:


## Dealing with the imbalance of the data using SVMOTE for oversampling. 
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import RandomOverSampler

oversample = SVMSMOTE(random_state = 42)
oversample = RandomOverSampler(random_state=42)

X_train, y_train = oversample.fit_resample(X_train, y_train)


# In[64]:


from sklearn.tree import DecisionTreeClassifier

start = time.time()
model = DecisionTreeClassifier(max_depth = 8).fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)
end_predict = time.time()

# evaluate the model
log_scores("Decision Tree", y_test, y_predictions)
print("Decision Tree\n" + classification_report(y_test, y_predictions))


# In[65]:


from sklearn.ensemble import RandomForestClassifier

start = time.time()
model = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                               random_state=0, bootstrap=True).fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)
end_predict = time.time()

# evaluate the model
log_scores("Random Forest", y_test, y_predictions)


# In[66]:


print("Random Forest Model\n" + classification_report(y_test, y_predictions))


# In[67]:


from sklearn.ensemble import GradientBoostingClassifier

start = time.time()
model = GradientBoostingClassifier().fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)
end_predict = time.time()

# evaluate the model
log_scores("Gradient Boosting", y_test, y_predictions)
print("Gradient Boosting\n" + classification_report(y_test, y_predictions))


# In[68]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# create the model
knn = KNeighborsClassifier()

# define the parameter grid
param_grid = {'n_neighbors': range(2, 20)}

# create the grid search object
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# fit the grid search to the data
grid_search.fit(X_train, y_train)

# print the best parameters
print(grid_search.best_params_)
start = time.time()
model = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test) # predictions from the testset
end_predict = time.time()

# evaluate the model
log_scores("k-NN", y_test, y_predictions)
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_predictions, cmap=plt.cm.YlGnBu)


# In[69]:


from sklearn.naive_bayes import GaussianNB

start = time.time()
model = GaussianNB().fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)
end_predict = time.time()

# evaluate the model
log_scores("Gaussian Naive Bayes", y_test, y_predictions)
print("Gaussian Naive Bayes\n" + classification_report(y_test, y_predictions))
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_predictions, cmap=plt.cm.YlGnBu)  


# In[71]:


from sklearn.neural_network import MLPClassifier

start = time.time()
model = MLPClassifier(random_state=1, max_iter=600, learning_rate="invscaling").fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)
end_predict = time.time()

# evaluate the model
log_scores("Multi-layer Perceptron", y_test, y_predictions)
print("Multi-layer Perceptron\n" + classification_report(y_test, y_predictions))
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_predictions, cmap=plt.cm.YlGnBu)  


# In[72]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter = 1000)
log_reg.fit(X_train, y_train)

y_pred_test = log_reg.predict(X_test)
y_pred_train = log_reg.predict(X_train)

# evaluate the model
# calculate accuracy on the training set
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy:.2%}")

# calculate accuracy on the test set
lr_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Testing Accuracy: {lr_accuracy:.2%}")

print('Classification report:')
print(classification_report(y_test, y_pred_test))


# In[73]:


model_performance


# In[74]:


# Graphical represantion of models 
def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')
model_performance.style.apply(highlight_max, props='color:white;background-color:darkblue;', axis=0)\
         .apply(highlight_max, props='color:white;background-color:red;', axis=1)\
         .apply(highlight_max, props='color:white;background-color:purple', axis=None)


# In[ ]:




