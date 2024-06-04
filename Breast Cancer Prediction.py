#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


df=pd.read_csv('data.csv')


# In[6]:


df.head(10)


# In[7]:


df.info()


# In[9]:


df.isna().sum()


# In[10]:


df.describe()


# In[13]:


df=df.dropna(axis=1)


# In[14]:


df.head(10)


# In[15]:


df.shape


# In[16]:


df['diagnosis'].value_counts()


# In[20]:


import seaborn as sns


# In[23]:


sns.countplot(df['diagnosis'],label="count")


# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[25]:


lb=LabelEncoder()


# In[28]:


def diagnosis_value(diagnosis): 
    if diagnosis == 'M': 
        return 1
    else: 
        return 0
  
df['diagnosis'] = df['diagnosis'].apply(diagnosis_value) 


# In[29]:


sns.lmplot(x='radius_mean',y='texture_mean',hue='diagnosis',data=df)


# In[30]:


df.head(10)


# In[31]:


df['diagnosis'].value_counts()


# In[32]:


df.corr()


# In[33]:


import matplotlib.pyplot as plt


# In[34]:


plt.figure(figsize=(10,10))


# In[39]:


sns.heatmap(df.iloc[:,1:10].corr(),annot=True)


# In[41]:


sns.pairplot(df.iloc[:,1:5],hue="diagnosis")


# In[43]:


X=df.iloc[:,2:32].values


# In[53]:


y=df.iloc[:,1].values
y


# In[46]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[49]:


from sklearn.preprocessing import StandardScaler


# In[50]:


st=StandardScaler()


# In[51]:


X_train=st.fit_transform(X_train)
X_test=st.fit_transform(X_test)


# In[52]:


X_train


# In[54]:


X_train.shape


# In[55]:


y_train.shape


# In[60]:


from sklearn.linear_model import LogisticRegression,LinearRegression


# In[57]:


log=LogisticRegression()


# In[58]:


log.fit(X_train,y_train)


# In[61]:


log.score(X_train,y_train)


# In[65]:


from sklearn.metrics import accuracy_score,classification_report


# In[66]:


accuracy_score(y_test,log.predict(X_test))


# In[68]:


print(classification_report(y_test,log.predict(X_test)))


# In[69]:


import pickle


# In[70]:


pickle.dump(log,open("model.pkl","wb"))


# In[71]:


pickle.load(open("model.pkl","rb"))


# In[ ]:




