# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:25:32 2020

@author: Shrikant Agrawal
"""

import pandas as pd   # Read data and data reprocessing
import numpy as np    # Work with arrays - multidimentional and single dimentional
import matplotlib.pyplot as plt  #Visualization
import seaborn as sns           # Visualization and statistical concept which help to visualize

train = pd.read_csv('titanic_train.csv')

# Missing Data - we can create heatmap by using seaborn to get missing values

train.isnull()   # It will show o/p as True or False and it is difficult to get NaN values

sns.heatmap(train.isnull())

#If you want to remove Y axis labels and cbar which showing values from 0-1 and change color
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap='viridis')

# Now deepdive into the dataset. To view data in particular column
sns.countplot(x='Survived',data=train)

# To set grid line in the above commnad
sns.set_style(style='whitegrid')
sns.countplot(x='Survived',data=train)

# Now you know how many people survived, further to divide it between male and female
sns.set_style(style ='whitegrid')
sns.countplot(x='Survived', hue='Sex',data=train)

# Now check how many passendgers survived in Passenger class
sns.set_style(style ='whitegrid')
sns.countplot(x='Survived', hue='Pclass',data=train, palette = 'rainbow')

""" By looking at the above graph we can say passenger class 1 ie richer people died 
less compare to passenger class 2 and 3. Most of the pclass 1 people survived."""

# To see what age group people are sailing in the boat  (drop missing values)
sns.distplot(train['Age'].dropna())

#To remove curve line ie kernal density
sns.distplot(train['Age'].dropna(),kde=False, color='darkred',bins=40)

# Alternatively we can do above operation by usning hist
train['Age'].hist(bins=30, color='darkred', alpha=0.5)  #aplha to reduce darkness

sns.countplot(x='SibSp',data=train)

train['Fare'].hist(bins=40, color='green',figsize=(8,4))

#Check any relation between Pclass and age
# plt.figure(figsize=(12,7))  to increase output size
sns.boxplot(x='Pclass', y='Age',data=train, palette='winter')

""" We have many missing values in the column Age to replace missing values, here we
have seen some relaition between Age column and PClass.
Mean value for PClass 1 ie Q2 is 37, for PClass2 and 3 its 29 and 24 respe.
Lets replace missing values based on Pclass"""

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 29
        
        else:
            return 24
    else:
        return Age

train['Age']=train[['Age', 'Pclass']].apply(impute_age,axis=1)
        
# Now lets check the heatmap again missing values in colum Age got replace
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


""" Now we can see missing values got replaced for Age column but theire are lot of
missing values in column cabin to replace it we need to do feature engineering which 
involves lot of logic. We will see it later, hence now we will drop this column"""

train.drop('Cabin', axis=1, inplace=True)

# Now lets check the heatmap again
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)

# Now convert categorical variable into numeric
train.info()

embark = pd.get_dummies(train['Embarked'],drop_first=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)

# Now we don't require below column
train.drop(['Sex','Embarked', 'Name','Ticket'],axis =1, inplace=True)
train.head()

# Now include two columns which we have created now ie sex and embark
train = pd.concat([train,embark,sex], axis=1)
train.head()

# Our final data is ready
"""Build a Logistic Regression Model. First divide data into train and test"""
# Divide the data between dependent variable ie Survived and indepdent variable ie rest all columns
x= train.drop('Survived', axis=1)

y = train['Survived']

# Split Train Test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=101)

# Initialize Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

# Apply model on our test dataset
prediction = logmodel.predict(x_test)

# To check accuracy of our model  - use confusion matrix
from sklearn.metrics import confusion_matrix
accuracy = confusion_matrix(y_test, prediction)
accuracy

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)
accuracy

""" To increase accuracy of our model we can apply xgboost also"""
