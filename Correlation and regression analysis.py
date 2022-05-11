#!/usr/bin/env python
# coding: utf-8

# In[58]:


from sklearn import datasets, svm, metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


# In[59]:


data = load_boston()


# In[60]:


data["data"]


# In[61]:


df = pd.DataFrame(data["data"], columns = data['feature_names'])


# In[32]:


df.head()


# In[62]:


df["MEDV"] = data["target"]


# In[34]:


df.head()


# #### Correlation visualisation

# In[35]:


df.plot(kind = "scatter", x = "MEDV", y = "CRIM")


# In[36]:


# Graphics for all of the values in the dataset
sns.pairplot(df)


# In[37]:


df.plot(kind = "scatter", x = "LSTAT", y = "RM")


# In[46]:


df["Price per m2"] = df["MEDV"]/df["RM"]


# In[39]:


df.head()


# In[41]:


df = df["Price per m2"].fillna(0)


# In[47]:


#Plot of the conne tion between the value of the property and price of one room
df.plot(kind = "scatter", x = "MEDV", y = "Price per m2")


# In[48]:


#Pearson correlation coefficient
df[["MEDV","Price per m2"]].corr()


# In[49]:


#Spearman correlation coefficient
df[["MEDV","Price per m2"]].corr(method = "spearman")


# In[50]:


#Kendall correlation coefficient
df[["MEDV","Price per m2"]].corr(method = "kendall")


# In[51]:


df.corr()


# In[53]:


# annot is for writing down the value in the cell if True
sns.set(rc = {'figure.figsize' : (11.8,8.27)})
sns.heatmap(df.corr(), annot = True)


# In[54]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[65]:


X = df[["RM"]]
y = df["MEDV"]


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.30, random_state = 42 )


# In[68]:


X_train.shape


# In[69]:


model = LinearRegression()


# In[70]:


model.fit(X_train, y_train) #model tries to find optimal coefficients


# In[72]:


# a
model.coef_


# In[73]:


# b
model.intercept_


# In[74]:


y_pred = model.predict(X_test)


# In[75]:


y_pred


# In[76]:


y_test


# In[77]:


model.score(X_test, y_test) #finding out the determination coef


# In[78]:


plt.scatter(X_test, y_test)


# In[79]:


plt.plot(X_test, y_pred, c = "r")


# In[80]:


plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, c = "r")


# In[82]:


plt.scatter(y_test, y_pred) #same one


# In[ ]:




