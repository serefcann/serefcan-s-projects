#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


# In[37]:


titanic = pd.read_csv('C:\\Users\\şerefcanmemiş\\Documents\\DATAS\\tested.csv')


# In[38]:


titanic.head()


# In[39]:


titanic.isna().sum()


# In[40]:


titanic.shape


# In[41]:


titanic.info()


# In[42]:


columns= ['Age','Fare']
for col in columns:
    titanic[col].fillna(titanic[col].median(),inplace=True)
titanic['Cabin'].fillna('Unknown',inplace=True)
titanic.isna().sum()


# In[43]:


dup = titanic.duplicated().sum()
print('We have {} duplicated values'.format(dup))


# In[44]:


titanic.head()


# In[45]:


titanic['Title'] = titanic['Name'].str.extract(r',\s(.*?)\.')

titanic['Title'] = titanic['Title'].replace('Ms', 'Miss')
titanic['Title'] = titanic['Title'].replace('Dona', 'Mrs')
titanic['Title'] = titanic['Title'].replace(['Col', 'Rev', 'Dr'], 'Rare')


# In[46]:


titanic.head()


# In[47]:


bins = [-np.inf, 17, 32, 45, 50, np.inf]
labels = ["Children", "Young", "Mid-Aged", "Senior-Adult", 'Elderly']
titanic['Age_Group'] = pd.cut(titanic['Age'], bins = bins, labels = labels)


# In[48]:


titanic.head()


# In[49]:


titanic['family'] = titanic['SibSp'] + titanic['Parch']


# In[50]:


titanic.head()


# In[51]:


titanic.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[52]:


titanic.head()


# In[53]:


col_to_move=titanic.pop('Age_Group')
titanic.insert(4,'Age_Group',col_to_move)
col_to_move2=titanic.pop('family')
titanic.insert(7,'family',col_to_move2)


# In[54]:


titanic.head()


# In[55]:


titanic['Age_Group']=titanic['Age_Group'].astype('object')


# In[56]:


titanic['Age_Group']


# In[57]:


titanic.describe()


# In[58]:


titanic.describe(include='O')


# In[59]:


titanic.groupby('Sex')[['Survived','Pclass','Age','SibSp','Parch','family','Fare']].mean()


# In[60]:


titanic.groupby('Embarked')[['Survived','Pclass','Age','SibSp','Parch','family','Fare']].mean()


# In[61]:


titanic.groupby('Age_Group')[['Survived','Pclass','Age','SibSp','Parch','family','Fare']].mean()


# In[62]:


titanic.head()


# In[63]:


from sklearn.preprocessing import LabelEncoder
cols=['Sex','Age_Group','Cabin','Embarked','Title']
le=LabelEncoder()
for col in cols:
    titanic[col]=le.fit_transform(titanic[col])


# In[64]:


titanic.Survived.value_counts()


# In[65]:


colors='r','b'
labels=0,1
survived_count=titanic['Survived'].value_counts()
plt.pie(survived_count,labels=labels,colors=colors)
plt.show()


# In[66]:


titanic.head()


# In[67]:


X=titanic.drop('Survived',axis=1)
Y=titanic['Survived']


# In[ ]:





# In[68]:


smote = SMOTE(random_state=42)
X_balanced , Y_balanced = smote.fit_resample(X,Y)


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(X_balanced, Y_balanced, test_size = 0.3, random_state = 42) 


# In[74]:


sc=StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[75]:


lr = LogisticRegression()
rf = RandomForestClassifier()
gbc = GradientBoostingClassifier()

lr.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
gbc.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
rf_pred = rf.predict(X_test_scaled)
gbc_pred = gbc.predict(X_test_scaled)


# In[76]:


lr_report = classification_report(y_test, lr_pred)
lr_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='accuracy')

rf_report = classification_report(y_test, rf_pred)
rf_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='accuracy')

gbc_report = classification_report(y_test, gbc_pred)
gbc_scores = cross_val_score(gbc, X_train_scaled, y_train, cv=5, scoring='accuracy')


print('The classification report of Logistic Regression is below : ', '\n\n\n', lr_report)
print(f"Logistic Regression Mean Cross-Validation Score: {lr_scores}")

print('\n', '='*100, '\n')
print('The classification report of Random Forest is below : ', '\n\n\n', rf_report)
print(f"Random Forest Mean Cross-Validation Score: {rf_scores}")

print('\n', '='*100, '\n')
print('The classification report of Gradient Bossting Classifier is below : ', '\n\n\n', rf_report)
print(f"Gradient Boosting Classifier Mean Cross-Validation Score: {gbc_scores}")


# In[ ]:





# In[ ]:





# In[ ]:




