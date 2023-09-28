#!/usr/bin/env python
# coding: utf-8

# # Feature Extraction and Selection Final Project
# ## Problem : Implement the SFS using wrapper method to find best subet of features.
# ### Student name: Sohrab Pirhadi
# ### Student Number: 984112

# ## Importing Libraries

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt


# ### Reading Dataset

# In[2]:


def readData(fname, pos_label):
    with open(fname,'r') as ifile:

        feat_vecs=[]
        labels=[]
        genes=[]
        
    
        for ln in ifile: #ifile is a 103*2135 dataset
            ln = ln.strip()
           
            ln=ln.split(' ') 
            if ln[0][0]=='y':
                for l in ln[1:]: 
                    if l==pos_label:
                        labels+=[1]
                    else:
                        labels+=[0]
            else:    
                vector=[]
                for f in ln[1:]:
                    vector+=[np.float(f)]
                feat_vecs+=[vector]
                genes+=[ln[0]]            
            
        return np.array(feat_vecs).T, np.array(labels), genes


# In[3]:


fname='prostate_preprocessed.txt'
X,Y, genes=readData(fname, 'tumor')


# In[4]:


X.shape


# ### Rows are samples(102)

# In[5]:


X.shape[0]


# In[6]:


X[0] #first row values(all features values for fisrt sample)


# ### Columns are features(2135)

# In[7]:


len(genes)


# In[8]:


X.shape[1]


# In[9]:


X[:,0] # first column values(first feature value for all samples)


# ### Labels(102) 

# In[10]:


Y # 0 for Normal and 1 for tumor


# ### splitting data into training and testing

# In[11]:


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=101) #dividing with train test split
y_train.shape


# In[12]:


X_test[:,0:1]


# In[13]:


X_test[:,0].reshape(-1,1)


# ## ML algorithm used --> KNN

# In[14]:


knn = KNeighborsClassifier(n_neighbors=2) 


# ## Sequentioal Forward Selection 

# In[15]:


kFeatures = 5
featureSubset = []
remainingFeatures = [x for x in  range(0,X.shape[1])]
maxScore = 0
maxIndex = -1
for i in range(len(remainingFeatures)):
        featureSubset = []
        if(i not in featureSubset):
            featureSubset.append(i)
        else:
            continue
        knn = KNeighborsClassifier()
        knn.fit(X_train[:,featureSubset].reshape(-1,1) , y_train)
        predict = knn.predict(X_test[:,i].reshape(-1,1))
        currentScore = f1_score(y_test, predict)
        if currentScore > maxScore:
            maxScore = currentScore
            maxIndex = i
        featureSubset.pop()
featureSubset.append([maxIndex,maxScore])
print(featureSubset)


    


# In[16]:


while len(featureSubset) < kFeatures:
    maxScore = 0
    maxIndex = -1
    tempSubset = [x[0] for x in featureSubset]
    for i in range(len(remainingFeatures)):
        if(i not in tempSubset):
            tempSubset.append(i)
        else:
            continue
        knn = KNeighborsClassifier()
        knn.fit(X_train[:,tempSubset] , y_train)
        predict = knn.predict(X_test[:,tempSubset])
        currentScore = f1_score(y_test, predict)
        if currentScore > maxScore:
            maxScore = currentScore
            maxIndex = i
        tempSubset.pop()
    featureSubset.append([maxIndex,maxScore])
    print(featureSubset)
    
        


# ### Plotting The Reults

# In[26]:


f_score = [item[1] for item in featureSubset]
f_score


# In[31]:


plt.plot([1,2,3,4,5],f_score, 'ro')
plt.xlabel('Number of Features')
plt.ylabel('F1-Score')


# ## Selected Genes

# In[28]:


features_index = [item[0] for item in featureSubset]
features_index


# In[29]:


for index in features_index:
    print(genes[index])


# In[ ]:




