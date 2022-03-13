# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 09:23:30 2022

@author: LENOVO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel("datst.xlsm")


df.drop("User ID", axis =1 , inplace = True)

df.drop("Name",axis =1, inplace=True)

df.drop("Phone", axis=1,inplace=True)

X = df.iloc[:,:-1].values
y = df.iloc[:,3].values


X.reshape(-1,1)
y.reshape(-1,1)


from sklearn.preprocessing import LabelEncoder
labelencoder_gender = LabelEncoder()
X[:,0] = labelencoder_gender.fit_transform(X[:,0])

X = np.vstack(X[:,:]).astype(float)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0,solver="liblinear")
classifier.fit(X_train,y_train)'''


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)



classifier.predict(X_test)


import pickle 
with open("Random.pkl",'wb',) as f:
    pickle.dump(classifier,f)



y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred,y_test)
print(acc)


x2 = classifier.predict([[1,19,19000]])
x3 = classifier.predict([[0,19,19000]])
x4 = classifier.predict([[1,60,42000]])


print("Male and prediction is : ", x2)
print("Female and prediction is : " ,x3)
print("Female and prediction is : " , x4)