
# coding: utf-8

# In[12]:

#importing required libraries
import pandas as pd
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
import pickle


# In[13]:

# reading in the data
dataset = pd.read_csv('F:\\haloween_data\\test.csv')
print dataset


# In[14]:

#data preprocessing
le = preprocessing.LabelEncoder()
dataset.color = pd.Series(le.fit_transform(dataset.color),dtype = 'category')


# In[15]:

#Creating independent and dependent variables
X = dataset
X = X.drop('id',1)


# In[16]:

filename = 'C:\Users\Vignesh\Documents\\ghost_dt.sav'
clf_model = pickle.load(open(filename, 'rb'))


# In[17]:

#predicting training and testing models
Y = clf_model.predict(X)


# In[18]:

#Calculaitng accuray
type = pd.DataFrame({'type':Y})
final_dataset = dataset.join(type)


# In[19]:

final_dataset.to_csv(path_or_buf = 'C:\Users\Vignesh\Downloads\\ghost_dt_pred.csv')


# In[ ]:



