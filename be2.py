# for numerical computing
import numpy as np

# for datatrames
import pandas as pd

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

 

# to split train and test set
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import plot_roc_curve, recall_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

 

 

   

# to save the final model on disk

df=pd.read_csv('Autism traits data.csv')
print(df)
print(df.shape)
print(df.columns)

print(df.head())

print(df.describe())
print(df.corr())

df = df.drop_duplicates()
print( df.shape )

print (df.isnull().sum())
df=df.dropna()

print (df.isnull().sum())

gender = {'m': 1,'f': 0}

df.gender = [gender[item] for item in df.gender]

bornwithjaundice = {'yes': 1, 'no': 0}
df.bornwithjaundice = [bornwithjaundice[item] for item in df.bornwithjaundice]

familymemberswithautism = {'yes': 1,'no': 0}

df.familymemberswithautism = [familymemberswithautism[item] for item in df.familymemberswithautism]

 

usedscreeningbefore= {'yes': 1,'no': 0}

df.usedscreeningbefore = [usedscreeningbefore[item] for item in df.usedscreeningbefore]

   

Whoistakingthetest= {'Self': 1, 'Parent': 2, 'Health care professional':3, 'Relative':4,'Others':5}

df.Whoistakingthetest = [Whoistakingthetest [item] for item in df.Whoistakingthetest]

 

HasAutism= {'YES': 1,'NO': 0}
df.HasAutism = [HasAutism[item] for item in df.HasAutism]
print (df)

from sklearn import preprocessing
# Typecasting
df["race"] = df["race"].astype(str)
df["placeofresidence"] = df["placeofresidence"].astype(str)
df["agedesc"] = df["agedesc"].astype(str)
df["Whoistakingthetest"] = df["Whoistakingthetest"].astype(str)

# Initializing Encoder
number = preprocessing.LabelEncoder()

# Encoding
df["race"] = number.fit_transform(df["race"])
df["placeofresidence"] = number.fit_transform(df["placeofresidence"])
df["agedesc"] = number.fit_transform(df["agedesc"])
df["Whoistakingthetest"] = number.fit_transform(df["Whoistakingthetest"])

df.head()
y = df.HasAutism
print (y)
#X=df[['age', 'gender', 'bornwithjaundice', 'familymemberswithautism', 'usedscreeningbefore', 'Whoistakingthetest','q1','q2','q3','q4','q5','q6','q7','q8','q9','q10']]
X = df.drop(["HasAutism"], axis = 1).values
print(X)


# Split X and y into train and test sets,
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Print number of observations in X_train, X_test, y_train, and y_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#ML models
model1= LogisticRegression()
model2=RandomForestClassifier(n_estimators=500)
model3=svm.SVC(kernel = 'linear',C= 1)
model4= KNeighborsClassifier(n_neighbors=5)
model5=DecisionTreeClassifier()

#Train ML models with trained data
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)


#2 Predict Test set results
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)


acc = accuracy_score(y_test, y_pred1) ## get the accuracy on testing data
print("Accuracy of Logistic Regression is {:.2f}%".format(acc*100))


acc = accuracy_score(y_test, y_pred2) ## get the accuracy on testing date
print("Accuracy of RandomForestClassifier is {:.2f}%".format(acc*100))

acc = accuracy_score(y_test, y_pred3) ## get the accuracy on testing date
print("Accuracy of SVM is {:.2f}%".format(acc*100))

acc = accuracy_score(y_test, y_pred4) ## get the accuracy on testing data
print ("Accuracy of KNieighborsClassifier is {:.2f}%".format(acc*100))

 
acc = accuracy_score(y_test, y_pred5) ## get the accuracy on testing data
print ("Accuracy of Decision Tree is {:.2f}%".format(acc*100))


import joblib
joblib.dump(model3, 'ASD_final.pkl')
final_model =joblib.load('ASD_final.pkl')

pred=final_model.predict(X_test)

acc = accuracy_score(y_test, pred) ## get the accuracy on testing data
print("Final Model Accuracy is {:.2f}%".format(acc*100))
from sklearn.metrics import classification_report
print("Classification Report for Logistic Regression classifier:\n",classification_report(y_test,y_pred1))
print("Classification Report for Random Forest classifier:\n",classification_report(y_test,y_pred2))
print("Classification Report for SVM classifier:\n",classification_report(y_test,y_pred3))
print("Classification Report for KNearest Neighbors classifier:\n",classification_report(y_test,y_pred4))
print("Classification Report for Decision tree classifier:\n",classification_report(y_test,y_pred5))

#Logistic Regression accuracy and recall scores for train and test data

plot_roc_curve(model1,X_train,y_train)
y_train_pred = model1.predict(X_train)

print("Logistic Regression train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("Logistic Regression train recall score: ",recall_score(y_train,y_train_pred))

plot_roc_curve(model1,X_test,y_test)
y_test_pred = model1.predict(X_test)

print("Logistic Regression test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("Logistic Regression test recall score: ",recall_score(y_test,y_test_pred)) 

#Random Forest classifier accuracy and recall scores for train and test data

plot_roc_curve(model2,X_train,y_train)
y_train_pred = model2.predict(X_train)

print("Random Forest classifier train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("Random Forest classifier train recall score: ",recall_score(y_train,y_train_pred))

plot_roc_curve(model2,X_test,y_test)
y_test_pred = model2.predict(X_test)

print("Random Forest classifier test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("Random Forest classifier test recall score: ",recall_score(y_test,y_test_pred)) 



#SVM accuracy and recall scores for train and test data

plot_roc_curve(model3,X_train,y_train)
y_train_pred = model3.predict(X_train)

print("SVM classifier train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("SVM classifier train recall score: ",recall_score(y_train,y_train_pred))

plot_roc_curve(model3,X_test,y_test)
y_test_pred = model3.predict(X_test)

print("SVM classifier test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("SVM classifier test recall score: ",recall_score(y_test,y_test_pred)) 


#KNN classifier accuracy and recall scores for train and test data

plot_roc_curve(model4,X_train,y_train)
y_train_pred = model4.predict(X_train)

print("KNN classifier train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("KNN classifier train recall score: ",recall_score(y_train,y_train_pred))

plot_roc_curve(model4,X_test,y_test)
y_test_pred = model4.predict(X_test)

print("KNN classifier test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("KNN classifier test recall score: ",recall_score(y_test,y_test_pred)) 

 
#Decision tree classifier accuracy and recall scores for train and test data

plot_roc_curve(model5,X_train,y_train)
y_train_pred = model5.predict(X_train)

print("Decision tree classifier train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("Decision tree classifier train recall score: ",recall_score(y_train,y_train_pred))

plot_roc_curve(model5,X_test,y_test)
y_test_pred = model5.predict(X_test)

print("Decision tree classifier test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("Decision tree classifier test recall score: ",recall_score(y_test,y_test_pred)) 







 

 

