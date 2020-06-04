#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

train=pd.read_csv('https://raw.githubusercontent.com/Prabhutva711/textclass/master/SPAM%20text%20message%2020170820%20-%20Data.csv')
# In[3]:


train


# In[4]:


import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
ps=PorterStemmer()


# In[5]:


def remove_punc(mess):
    rt=[]
    fn2=[]
    fn1=[]
    
    for x in mess:
        if x not in string.punctuation:
            rt.append(x)
    rt= ''.join(rt)
    rt=rt.split()
    for y in rt:
        fn1.append(y)
        
    for y in fn1:
        if y not in stopwords.words('english'):
            fn2.append(y)
    for z in fn2:
        z=z.ps.stem(z)
    
    return fn2


# In[6]:


train['Message']=train['Message'].apply(remove_punc)


# In[7]:


train['Length']=train['Message'].apply(len)


# In[8]:


train['Message']=train['Message'].apply(' '.join)


# In[9]:


def hs(mess):
    if mess=="ham":
        return 1
    else:
        return 0


# In[10]:


train['Category']=train['Category'].apply(hs)


# In[11]:


train.head(10)


# In[12]:


#creating bag of words


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
bow=cv.fit_transform(train['Message'])


# In[14]:


#converting bow to tfidf 


# In[15]:


from sklearn.feature_extraction.text import TfidfTransformer
bow = TfidfTransformer().fit_transform(bow)
bow=bow.toarray()
bow=bow.tolist()


# In[16]:


#concatenating length to bow


# In[62]:


for i in range(5572):
    bow[i].append(train['Length'].iloc[i])


# In[18]:


#splitting train_test


# In[65]:


from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(bow,train['Category'],test_size=0.3)


# TFIDF=term frequency inverse document frequency=> is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
# TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.
# 
# One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model.
# 
# Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
# 
# TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:
# 
# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
# 
# IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:
# 
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
# 
# See below for a simple example.

# Example:
# 
# Consider a document containing 100 words wherein the word cat appears 3 times.
# 
# The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.
# 
# 

# In[20]:


#lets see what Logistic Regression CLassifier Is And When To Use It


# [1]-Logistic Classifier is predominantly a binary classifier.
# 
# 

# In[22]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=2)
logreg_train = grid.fit(xtr,ytr)


# In[1]:


import cv2


# In[23]:


from sklearn.metrics import confusion_matrix
print (confusion_matrix(yte,logreg_train.predict(xte)))


# now lets try scale down train['Length'] to 0-1 scale

# In[26]:


min=[]
max=[]
for i in bow:
    min.append(i[-1])


# In[40]:


def scale(mess):
    return mess/86


# In[63]:


for i in bow:
    i[-1]=i[-1]/86


# now lets see what happens when we scale length

# In[46]:


from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(bow,train['Category'],test_size=0.3)


# In[47]:


logreg = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=2)
logreg_train = grid.fit(xtr,ytr)


# In[48]:


from sklearn.metrics import confusion_matrix
print (confusion_matrix(yte,logreg_train.predict(xte)))


# only resulted in not so good transition
# 
#MultinomialMB
# In[49]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(xtr,ytr)


# In[53]:


print (confusion_matrix(yte,spam_detect_model.predict(xte)))


# In[54]:


#length in now removal


# In[55]:


for i in bow:
    i.pop(-1)


# In[85]:


from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(bow,train['Category'],test_size=0.2)


# In[58]:


logreg = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(logreg, param_grid, cv=2)
logreg_train = grid.fit(xtr,ytr)


# In[59]:


print (confusion_matrix(yte,logreg_train.predict(xte)))


# MultiBinomial without length

# In[60]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(xtr,ytr)


# In[61]:


print (confusion_matrix(yte,spam_detect_model.predict(xte)))


# restoring Length back

# In[68]:


#LETS USE RANDOM FOREST CLASSIFIER ,ADA BOOST AND GRADIENT BOOST


# In[67]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
param_grid = {'n_estimators': [1, 100]}

grid = GridSearchCV(rfc, param_grid, cv=5)
rfc_best = grid.fit(xtr,ytr)
print (confusion_matrix(yte,rfc_best.predict(xte)))


# In[75]:


#with gini='entropy'


# In[74]:


rfc = RandomForestClassifier(n_estimators=100, criterion='entropy')
rfc = grid.fit(xtr,ytr)
print (confusion_matrix(yte,rfc.predict(xte)))


# In[69]:


#lets use adaboost


# In[71]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(xtr,ytr)
print(confusion_matrix(yte,clf.predict(xte)))


# In[72]:


#lets use gradientboost


# In[73]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(xtr,ytr)
print(confusion_matrix(yte,gbc.predict(xte)))


# In[76]:


#svc


# In[77]:


from sklearn import svm
clf1 = svm.SVC()
clf1.fit(xtr,ytr)
print (confusion_matrix(yte,clf1.predict(xte
                                       )))


# In[78]:


#using kmeans


# In[79]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=70)
neigh.fit(xtr,ytr)
print (confusion_matrix(yte,neigh.predict(xte
                                       )))


# In[ ]:


#using stochastic gradient


# In[87]:


from sklearn.linear_model import SGDClassifier
lf = SGDClassifier()
lf.fit(xtr,ytr)
print(confusion_matrix(yte,lf.predict(xte)))


# In[86]:


from sklearn.metrics import classification_report
print (classification_report(yte,lf.predict(xte)))

