
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn import neural_network
from sklearn import model_selection
from sklearn import metrics
import pickle


# In[2]:

dataset = pd.read_csv('F:\\haloween_data\\test.csv')


# In[3]:

X = dataset
color_dummies = pd.get_dummies(dataset.color, prefix='color', prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True).astype(int)


# In[4]:

X = X.join(color_dummies)


# In[5]:

X = X.drop('color',1)


# In[6]:

X = X.drop('id',1)


# In[12]:

filename = 'C:\Users\Vignesh\Documents\\ghost_NN.sav'
clf_model = pickle.load(open(filename, 'rb'))


# In[13]:

Y = clf_model.predict(X)
print Y


# In[14]:

type = pd.DataFrame({'type':Y})
final_dataset = dataset.join(type)


# In[15]:

final_dataset.to_csv(path_or_buf = 'C:\Users\Vignesh\Downloads\\ghost_nn1_pred.csv')


# In[ ]:



