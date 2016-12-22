
# coding: utf-8

# In[245]:

import pandas as pd
from sklearn import neural_network
from sklearn import model_selection
from sklearn import metrics
import pickle


# In[21]:

dataset = pd.read_csv('F:\\haloween_data\\train.csv')
print dataset


# In[22]:

color_dummies = pd.get_dummies(dataset.color, prefix='color', prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True).astype(int)
print color_dummies


# In[23]:

dataset = dataset.join(color_dummies)
print dataset


# In[24]:

dataset = dataset.drop('color',1)


# In[49]:

Y = dataset.type
X = dataset.drop('type',1)
X = X.drop('id',1)


# In[328]:

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.4, random_state = 1234)
clf = neural_network.MLPClassifier(hidden_layer_sizes=(500, 50), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.02, power_t=0.5, max_iter=100, shuffle=True, random_state=None, tol=0.001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
clf = clf.fit(X_train,Y_train)


# In[329]:

Y_train_pred = clf.predict(X_train)
Y_test_pred = clf.predict(X_test)
print Y_test_pred


# In[330]:

print "Training Accuracy: ",metrics.accuracy_score(Y_train_pred,Y_train)
print "Testing Accuracy: ",metrics.accuracy_score(Y_test_pred,Y_test)


# In[250]:

filename = 'C:\Users\Vignesh\Documents\\ghost_NN.sav'
pickle.dump(clf, open(filename,'wb'))


# In[ ]:



