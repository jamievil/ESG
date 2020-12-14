#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split


# In[3]:


data = pd.read_csv('rescaled_dataset.csv')


# In[4]:


data.head()


# In[ ]:





# In[5]:


#Setting up the x and y variables
#X is EO/EC
y = data.iloc[:, 3].values
X = data.iloc[:, 5:13].values


# In[19]:


#Preparing a table to gather descrpitive statistics
descriptive_data = data.drop(columns=["Unnamed: 0", "sub_id", "sub_count", 'gender'])


# In[20]:


#getting the descriptive statistics
descriptive_data.describe()


# In[44]:


#Training the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


# In[45]:


#Training the classifier
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini', max_depth=3)


# In[46]:


#Fitting the model to the training dataset
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[47]:


#Evaluating the model
eval_model = classifier.score(X_train, y_train)
print("Accuracy: ", eval_model)


# In[48]:


#Printing a confusion matrix showing incorrect vs correct predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[49]:


from sklearn import metrics


# In[50]:


#Defining the accuracy of the model
print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))


# In[35]:


feature_list = list(data.iloc[:, 5:13].columns)


# In[36]:


#Creating a feature importance variable 
feature_imp = pd.Series(classifier.feature_importances_, index=feature_list).sort_values(ascending=False)


# In[37]:


#A list of feature importances
feature_imp


# In[ ]:




