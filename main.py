### SMOTE Algprithm

#Importing necessary packages:
mport numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns

#Reading dataset Credit Card Fraud Detection taken from Kaggle:
df = pd.read_csv('creditcard.csv')

#Analyzing class distribution
lass_dist=df['Class'].value_counts()
print(class_dist)
print('\nClass 0: {:0.2f}%'.format(100 *loan_status_dist[0] / (class_dist[0]+class_dist[1])))
print('Class 1: {:0.2f}%'.format(100 *loan_status_dist[1] / (class_dist[0]+class_dist[1])))

""" 
Class 0: 99.83%

Class 1: 0.17%
"""

#Splitting data into train and test
X = df.drop(columns=['Time','Class'])
y = df['Class']

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=100,test_size=0.3,stratify=y)

#Evaluating results without SMOTE
model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print('Accuracy ',accuracy_score(y_test,pred))
print(classification_report(y_test,pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt='.2g')


print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy=0.5, k_neighbors=5, random_state=100)
X_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

#Results of SMOTE
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res.ravel())
predictions = lr.predict(x_test)

print('Accuracy ', accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='.2g')
