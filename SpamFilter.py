#!/usr/bin/env python
# coding: utf-8

# In[1]:


#impoerting the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from ntlk.corpus import stopwords
from ntlk.stem.porter import PorteStemmer


# In[6]:


data=pd.read_csv("E:\\NMDS\\spam.csv",encoding="latin")
data.head()


# In[7]:


data.info()


# In[8]:


data.isna().sum()


# In[10]:


data.rename({"v1":"label","v2":"text"},inplace=True,axis=1)
data.tail()


# In[12]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['label']=le.fit_transform(data['label'])


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_text,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[14]:


data.describe()


# In[16]:


data.shape


# In[18]:


import matplotlib.pyplot as plt
data['label'].value_counts().plot()(kind="bar",figsize(12,6)
plt.xticks(np.arange(2)),('Non spam','spam'),rotation=0);


# In[19]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model=fit.(X_train_res,y_train_res)


# In[21]:


from sklearn.tree import RandomForestClassifier
model1= RandomForestClassifier()
model1=fit.(X_train_res,y_train_res)


# In[ ]:


from sklearn.naive_bayes import RandomForestClassifier
model1= RandomForestClassifier()
model1=fit.(X_train_res,y_train_res)

