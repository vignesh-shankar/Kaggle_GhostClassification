
# coding: utf-8

# In[146]:

#importing required libraries
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
import pickle


# In[15]:

# reading in the data
dataset = pd.read_csv('F:\\haloween_data\\train.csv')
print dataset


# In[16]:

#data preprocessing
le = preprocessing.LabelEncoder()
dataset.color = pd.Series(le.fit_transform(dataset.color),dtype = 'category')


# In[96]:

#Creating independent and dependent variables
Y = dataset.type
X = dataset.drop('type',1)
X = X.drop('id',1)


# In[252]:

#splitting dataset and trainig the RF model
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.4, random_state = 1234)
#clf = ensemble.RandomForestClassifier(n_estimators=800, criterion='gini', max_depth=2, min_samples_split=1, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=2, max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=5, min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
clf = clf.fit(X_train,Y_train)


# In[253]:

#predicting training and testing models
Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)


# In[254]:

#Calculaitng accuray
print "Training Accuracy: ",metrics.accuracy_score(Y_train_pred,Y_train)
print "Testing Accuracy: ",metrics.accuracy_score(Y_test_pred,Y_test)


# In[255]:

#Exporting model to local file system
filename = 'C:\Users\Vignesh\Documents\\ghost_dt.sav'
pickle.dump(clf, open(filename,'wb'))


# In[ ]:



