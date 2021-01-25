#!/usr/bin/env python
# coding: utf-8

# ### Import Important Libraries

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


os.chdir('E:\\prasad\\practice\\My Working Projects\\Completed\\NLP\\SentimentText Classification for Restaurant Reviews using Machine Learning models')


# ### Perform Imports and Load Data

# In[3]:


df=pd.read_table('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# ### Cleaning the texts

# In[7]:


df['Review'][0]


# In[8]:


import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)


# In[9]:


corpus


# ### Creating the Bag of Words model

# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,1].values


# In[11]:


X


# In[12]:


y


# ### Data Split into Train,Test

# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ### Model Building

# In[14]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
model=MultinomialNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy_score(y_test,y_pred)


# In[15]:


confusion_matrix(y_test,y_pred)


# ### Model Evaluation

# #### Create Function For Model Evaluation

# In[16]:


def check_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('accuracy_score:',accuracy_score(y_test,y_pred),'\n')
    print('Confusion Matrix')
    print(confusion_matrix(y_test,y_pred),'\n')
    print('Classification Report')
    print(classification_report(y_test,y_pred))


# ### Check Accuracy_score by using different algorithms

# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC


# In[18]:


# LogisticRegression
check_model(LogisticRegression(),X_train,X_test,y_train,y_test)


# In[19]:


# RandomForestClassifier
check_model(RandomForestClassifier(),X_train,X_test,y_train,y_test)


# In[20]:


# KNeighborsClassifier
check_model(KNeighborsClassifier(),X_train,X_test,y_train,y_test)


# In[21]:


# DecisionTreeClassifier
check_model(DecisionTreeClassifier(),X_train,X_test,y_train,y_test)


# In[22]:


# MultinomialNB
check_model(MultinomialNB(),X_train,X_test,y_train,y_test)


# In[23]:


# SVC
check_model(SVC(),X_train,X_test,y_train,y_test)


# #### MultinomialNB is Predict Best Accuracy-accuracy_score: 0.74

# In[24]:


mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_pred=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred))


# In[25]:


cm=confusion_matrix(y_test,y_pred)


# In[26]:


sns.heatmap(cm,annot=True)
plt.show()


# ### Save Model in Pickle & Joblib

# In[32]:


import pickle,joblib


# In[33]:


pickle.dump(mnb,open('review_clf_pkl','wb'))


# In[34]:


joblib.dump(mnb,'review_clf_jbl')


# #### Load Pickle Model

# In[35]:


model_pkl=pickle.load(open('review_clf_pkl','rb'))
y_pred_pkl=model_pkl.predict(X_test)
print(accuracy_score(y_test,y_pred_pkl))


# #### Load Joblib Model

# In[36]:


model_jbl=joblib.load('review_clf_jbl')
y_pred_jbl=model_jbl.predict(X_test)
print(accuracy_score(y_test,y_pred_jbl))


# In[37]:


confusion_matrix(y_test,y_pred_jbl)


# In[ ]:




