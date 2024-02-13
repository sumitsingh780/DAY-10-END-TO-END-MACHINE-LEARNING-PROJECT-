#!/usr/bin/env python
# coding: utf-8

# # END TO END TOY PROJECT 

# In[1]:


import numpy as np
import pandas as pd 


# In[2]:


df = pd.read_csv("placement-dataset.csv")


# In[3]:


df 


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# # preprocessing 

# In[7]:


df= df.iloc[:,1:]


# In[8]:


df.head()


# # EDA 

# In[9]:


import matplotlib.pyplot as plt


# In[10]:


plt.scatter(df["cgpa"],df['iq'],c=df['placement'])
plt.show()


# # extract input and output column 

# In[11]:


X =df.iloc[:,0:2]
Y = df.iloc[:,-1]


# In[12]:


X


# In[13]:


Y


# In[14]:


Y.shape


# # TRAIN TEST SPLIT

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)


# In[17]:


X_train


# In[18]:


Y_train


# In[19]:


X_test


# In[20]:


Y_test


# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


scaler = StandardScaler()


# In[23]:


X_train = scaler.fit_transform(X_train)


# In[24]:


X_train


# In[25]:


X_test = scaler.transform(X_test)


# In[26]:


X_test


# # using logistic regression classifier for training

# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


clf = LogisticRegression()


# In[29]:


clf.fit(X_train,Y_train)


# # evaluate the model 

# In[40]:


y_pred = clf.predict(X_test)


# In[41]:


Y_test


# In[42]:


from sklearn.metrics import accuracy_score


# In[43]:


accuracy_score(Y_test,y_pred)


# In[52]:


get_ipython().system('pip install mlxtend')


# In[53]:


from mlxtend.plotting import plot_decision_regions


# In[56]:


plot_decision_regions(X_train, Y_train.values, clf=clf, legend=2)


# In[57]:


import pickle


# In[59]:


pickle.dump(clf,open("model.pkl",'wb'))


# In[ ]:




