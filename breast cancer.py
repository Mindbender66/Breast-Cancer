#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# The sklearn. metrics module implements several loss, score, and utility functions to measure classification performance. 
# Some metrics might require probability estimates of the positive class, confidence values, or binary decisions values.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[2]:


data=pd.read_csv("breast_cancer.csv")


# In[3]:


data


# * The "isnull" function gives that the no,of null values in every attribute 

# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# - The data set contains 683 rows and 10 columns
# - The dataset does not contain any missing values as it is cleaned data
# - The data types of the dataset :int(int64)

# In[6]:


data["Class"].value_counts()


# In[7]:


data["Class"].value_counts().plot.pie()
plt.show()


# - here class is the dependent variable
# - it has 2 types class 2 and class 4
# - class 2 : Benign
# - class 4 : Malignant

# In[8]:


data["Clump Thickness"].value_counts()


# In[9]:


data["Uniformity of Cell Size"].value_counts()


# In[10]:


data["Uniformity of Cell Shape"].value_counts()


# In[11]:


data["Marginal Adhesion"].value_counts()


# In[12]:


data["Single Epithelial Cell Size"].value_counts()


# In[13]:


data["Bare Nuclei"].value_counts()


# In[14]:


data["Bland Chromatin"].value_counts()


# In[15]:


data["Normal Nucleoli"].value_counts()


# In[16]:


data["Mitoses"].value_counts()


# In[17]:


data.describe()


# In[18]:


corelation=data.corr()


# In[19]:


corelation


# In[22]:


sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns
           ,annot=True,cmap='RdBu')
plt.show()


# In[23]:


data.groupby("Class").mean()["Clump Thickness"]


# * There is a difference between the bar graph and the pie graph 

# In[24]:


data.groupby("Class").mean()["Clump Thickness"].plot.bar()
plt.show()
data.groupby("Class").mean()["Clump Thickness"].plot.pie()
plt.show()


# In[91]:


data.groupby("Class").mean()["Uniformity of Cell Size"]


# In[92]:


data.groupby("Class").mean()["Uniformity of Cell Size"].plot.bar()
plt.show()
data.groupby("Class").mean()["Uniformity of Cell Size"].plot.pie()
plt.show()


# In[30]:


data.groupby("Class").mean()["Uniformity of Cell Shape"].plot.bar()
plt.show()
data.groupby("Class").mean()["Uniformity of Cell Shape"].plot.pie()
plt.show()


# In[93]:


data.groupby("Class").mean()["Marginal Adhesion"].plot.bar()
plt.show()
data.groupby("Class").mean()["Marginal Adhesion"].plot.pie()
plt.show()


# In[94]:


data.groupby("Class").mean()["Single Epithelial Cell Size"].plot.bar()
plt.show()
data.groupby("Class").mean()["Single Epithelial Cell Size"].plot.pie()
plt.show()


# In[95]:


data.groupby("Class").mean()["Bare Nuclei"].plot.bar()
plt.show()
data.groupby("Class").mean()["Bare Nuclei"].plot.pie()
plt.show()


# In[96]:


data.groupby("Class").mean()["Bland Chromatin"].plot.bar()
plt.show()
data.groupby("Class").mean()["Bland Chromatin"].plot.pie()
plt.show()


# In[97]:


data.groupby("Class").mean()["Normal Nucleoli"].plot.bar()
plt.show()
data.groupby("Class").mean()["Normal Nucleoli"].plot.pie()
plt.show()


# In[98]:


data.groupby("Class").mean()["Mitoses"].plot.bar()
plt.show()
data.groupby("Class").mean()["Mitoses"].plot.pie()
plt.show()


# In[99]:


x= data.iloc[:,0:-1].values
y = data.iloc[:,-1].values


# In[100]:


x


# In[101]:


y


# In[102]:


datax=data[["Clump Thickness", "Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]]


# In[103]:


datax


# In[105]:


x


# In[106]:


datay=data[["Class"]]


# In[107]:


datay


# In[43]:


y


# In[109]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,random_state=0)


# In[110]:


classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)


# In[111]:


y_hat = classifier.predict(x_test)


# In[112]:


y_hat


# In[48]:


con_mat = confusion_matrix(y_test, y_hat)
con_mat


# In[113]:


accuracy = (83+47) / (83+47 + 3+ 4)
accuracy


# In[114]:


accuracies = cross_val_score(estimator = classifier,X= x_train,y=y_train, cv = 20)


# In[115]:


accuracies


# In[116]:


print("Accuracy : {:.4f} %".format(accuracies.mean()*100))

