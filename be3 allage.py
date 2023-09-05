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
from sklearn.metrics import confusion_matrix

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

 

 

   

# to save the final model on disk
df=pd.read_csv('asdallagedata.csv')
#df=pd.read_csv('Autism traits data.csv')
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

df['race'] = df['race'].apply(lambda x: 'Others' if x == '0' else x)
df['race'] = df['race'].apply(lambda x: 'Others' if x == 'others' else x)
df['race'] = df['race'].apply(lambda x: 'Hispanic' if x == 'Latino' else x)

HasAutism= {'YES': 1,'NO': 0}
df.HasAutism = [HasAutism[item] for item in df.HasAutism]
print (df)

df.head()

#EDA or Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
#correlation
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)
plt.title('Heatmap of Variable Correlations for any age criteria')
plt.show()

#gender vs asd
plt.figure(figsize=(15,5))
sns.barplot(x='gender',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Gender')
plt.xlabel('Gender')
plt.ylabel('Autism Spectrum Disorder')
plt.show()

#race vs asd
plt.figure(figsize=(15,5))
sns.barplot(x='race',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs race')
plt.xlabel('race')
plt.ylabel('Autism Spectrum Disorder')
plt.show()

#age vs asd
plt.figure(figsize=(15,5))
sns.barplot(x='HasAutism',y='age',data=df)
plt.title('Autism Spectrum Disorder vs Age')
plt.ylabel('Age')
plt.xlabel('Autism Spectrum Disorder')
plt.show()

#jaundice vs asd
plt.figure(figsize=(15,5))
sns.barplot(x='bornwithjaundice',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Jaundice')
plt.xlabel('bornwithjaundice')
plt.ylabel('Autism Spectrum Disorder')
plt.show()

#any family member with asd vs autism
plt.figure(figsize=(15,5))
sns.barplot(x='familymemberswithautism',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Family member with ASD')
plt.xlabel('Family member with ASD')
plt.ylabel('Autism Spectrum Disorder')
plt.show()

#Whois taking test vs asd
plt.figure(figsize=(15,5))
sns.barplot(x='Whoistakingthetest',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs (Relation or Whoistakingthetest)')
plt.xlabel('Whoistakingthetest or relation')
plt.ylabel('Autism Spectrum Disorder')
plt.show()

#Used screening test before vs asd
plt.figure(figsize=(15,5))
sns.barplot(x='usedscreeningbefore',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Used screening test before')
plt.xlabel('used screening test before')
plt.ylabel('Autism Spectrum Disorder')
plt.show()

#Q1 vs ASD

plt.figure(figsize=(15,5))
sns.barplot(x='q1',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Question 1')
plt.ylabel('Question 1')
plt.xlabel('Autism Spectrum Disorder')
plt.show()

#Q2 vs ASD

plt.figure(figsize=(15,5))
sns.barplot(x='q2',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Question 2')
plt.ylabel('Question 2')
plt.xlabel('Autism Spectrum Disorder')
plt.show()

#Q3 vs ASD

plt.figure(figsize=(15,5))
sns.barplot(x='q3',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Question 3')
plt.ylabel('Question 3')
plt.xlabel('Autism Spectrum Disorder')
plt.show()

#Q4 vs ASD

plt.figure(figsize=(15,5))
sns.barplot(x='q4',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Question 4')
plt.ylabel('Question 4')
plt.xlabel('Autism Spectrum Disorder')
plt.show()


#Q5 vs ASD

plt.figure(figsize=(15,5))
sns.barplot(x='q5',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Question 5')
plt.ylabel('Question 5')
plt.xlabel('Autism Spectrum Disorder')
plt.show()


#Q9 vs ASD

plt.figure(figsize=(15,5))
sns.barplot(x='q9',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Question 9')
plt.ylabel('Question 9')
plt.xlabel('Autism Spectrum Disorder')
plt.show()

#Q10 vs ASD

plt.figure(figsize=(15,5))
sns.barplot(x='q10',y='HasAutism',data=df)
plt.title('Autism Spectrum Disorder vs Question 10')
plt.ylabel('Question 10')
plt.xlabel('Autism Spectrum Disorder')
plt.show()



#X,y values
y = df.HasAutism
print (y)
X=df[['age', 'gender', 'bornwithjaundice', 'familymemberswithautism', 'usedscreeningbefore', 'Whoistakingthetest','q1','q2','q3','q4','q5','q6','q7','q8','q9','q10']]
#X = df.drop(["HasAutism"], axis = 1).values
print(X)


# Split X and y into train and test sets,
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


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
joblib.dump(model3, 'ASDallage_final.pkl')
final_model =joblib.load('ASDallage_final.pkl')

pred=final_model.predict(X_test)

acc = accuracy_score(y_test, pred) ## get the accuracy on testing data
print("Final Model Accuracy is {:.2f}%".format(acc*100))
from sklearn.metrics import classification_report
print("Classification Report for Logistic Regression classifier:\n",classification_report(y_test,y_pred1))
print("Classification Report for Random Forest classifier:\n",classification_report(y_test,y_pred2))
print("Classification Report for SVM classifier:\n",classification_report(y_test,y_pred3))
print("Classification Report for KNearest Neighbors classifier:\n",classification_report(y_test,y_pred4))
print("Classification Report for Decision tree classifier:\n",classification_report(y_test,y_pred5))

#confusion matrix
#LR
cmlr=confusion_matrix(y_pred1, y_test)
#Plot the confusion matrix.
sns.heatmap(cmlr,
            annot=True,
            fmt='g',
            xticklabels=['YES','NO'],
            yticklabels=['YES','NO'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix for Logistic regression',fontsize=17)
plt.show()

#RF
cmrf=confusion_matrix(y_pred2, y_test)
#Plot the confusion matrix.
sns.heatmap(cmrf,
            annot=True,
            fmt='g',
            xticklabels=['YES','NO'],
            yticklabels=['YES','NO'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix for Random forest Classifier',fontsize=17)
plt.show()

#SVM
cmsvm=confusion_matrix(y_pred3, y_test)
#Plot the confusion matrix.
sns.heatmap(cmsvm,
            annot=True,
            fmt='g',
            xticklabels=['YES','NO'],
            yticklabels=['YES','NO'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix for SVM classifier',fontsize=17)
plt.show()

cmknn=confusion_matrix(y_pred4, y_test)
#KNN confusion matrix.
sns.heatmap(cmknn,
            annot=True,
            fmt='g',
            xticklabels=['YES','NO'],
            yticklabels=['YES','NO'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix for KNN classifer',fontsize=17)
plt.show()

cmdt=confusion_matrix(y_pred5, y_test)
#DT confusion matrix.
sns.heatmap(cmdt,
            annot=True,
            fmt='g',
            xticklabels=['YES','NO'],
            yticklabels=['YES','NO'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix for Decision Tree classifier',fontsize=17)
plt.show()

#Roc curve

from sklearn.metrics import roc_curve, auc

# Train the SVM classifier
#classifier = svm.SVC(kernel='linear', probability=True, random_state=0)
y_score = model3.decision_function(X_test)

# Compute the ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
print("SVM roc curve:\n")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve for SVM model(area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for SVM Classifier')
plt.legend(loc="lower right")
plt.show()

print("LR roc curve:\n")

y_score = model1.decision_function(X_test)

# Compute the ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve for LR model (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Logistic Regression')
plt.legend(loc="lower right")
plt.show()

print("RF roc curve:\n")


# Predict probabilities for the positive class
y_scores = model2.predict_proba(X_test)[:, 1]

# Compute the false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Compute the AUC score
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve for RF classifier (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Random Forest classifier')
plt.legend(loc="lower right")
plt.show()


print("KNN roc curve:\n")

# Predict probabilities for the positive class
y_scores = model4.predict_proba(X_test)[:, 1]

# Compute the false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Compute the AUC score
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve for KNN classifier (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for KNN classifier ')
plt.legend(loc="lower right")
plt.show()


print("DT roc curve:\n")

# Predict probabilities for the positive class
y_scores = model5.predict_proba(X_test)[:, 1]

# Compute the false positive rate, true positive rate, and threshold
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Compute the AUC score
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f) for DT classifier ' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Decision tree classifier ')
plt.legend(loc="lower right")
plt.show()



#all metrics
#for train data
from sklearn import metrics
y_train_pred3= model3.predict(X_train)
print("SVM metrics:\n")
print("accuracy: ", metrics.accuracy_score(y_train,y_train_pred3))
print("precision: ", metrics.precision_score(y_train,y_train_pred3)) 
print("recall: ", metrics.recall_score(y_train,y_train_pred3))
print("f1: ", metrics.f1_score(y_train,y_train_pred3))
print("area under curve (auc): ", metrics.roc_auc_score(y_train,y_train_pred3))



#Logistic Regression accuracy and recall scores for train and test data

#plot_roc_curve(model1,X_train,y_train)
y_train_pred = model1.predict(X_train)

print("Logistic Regression train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("Logistic Regression train recall score: ",recall_score(y_train,y_train_pred))

#plot_roc_curve(model1,X_test,y_test)
y_test_pred = model1.predict(X_test)

print("Logistic Regression test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("Logistic Regression test recall score: ",recall_score(y_test,y_test_pred)) 

#Random Forest classifier accuracy and recall scores for train and test data

#plot_roc_curve(model2,X_train,y_train)
y_train_pred = model2.predict(X_train)

print("Random Forest classifier train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("Random Forest classifier train recall score: ",recall_score(y_train,y_train_pred))

#plot_roc_curve(model2,X_test,y_test)
y_test_pred = model2.predict(X_test)

print("Random Forest classifier test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("Random Forest classifier test recall score: ",recall_score(y_test,y_test_pred)) 



#SVM accuracy and recall scores for train and test data

#plot_roc_curve(model3,X_train,y_train)
y_train_pred = model3.predict(X_train)

print("SVM classifier train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("SVM classifier train recall score: ",recall_score(y_train,y_train_pred))

#plot_roc_curve(model3,X_test,y_test)
y_test_pred = model3.predict(X_test)

print("SVM classifier test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("SVM classifier test recall score: ",recall_score(y_test,y_test_pred)) 


#KNN classifier accuracy and recall scores for train and test data

#plot_roc_curve(model4,X_train,y_train)
y_train_pred = model4.predict(X_train)

print("KNN classifier train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("KNN classifier train recall score: ",recall_score(y_train,y_train_pred))

#plot_roc_curve(model4,X_test,y_test)
y_test_pred = model4.predict(X_test)

print("KNN classifier test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("KNN classifier test recall score: ",recall_score(y_test,y_test_pred)) 

 
#Decision tree classifier accuracy and recall scores for train and test data

#plot_roc_curve(model5,X_train,y_train)
y_train_pred = model5.predict(X_train)

print("Decision tree classifier train accuracy score: ",accuracy_score(y_train,y_train_pred))
print("Decision tree classifier train recall score: ",recall_score(y_train,y_train_pred))


#plot_roc_curve(model5,X_test,y_test)
y_test_pred = model5.predict(X_test)

print("Decision tree classifier test accuracy score: ",accuracy_score(y_test,y_test_pred))
print("Decision tree classifier test recall score: ",recall_score(y_test,y_test_pred)) 

"""#scaling for models


from sklearn.preprocessing import StandardScaler


# Perform standard scaling on the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaled models:\n")
# Logistic Regression
logreg = model1
logreg.fit(X_train_scaled, y_train)
logreg_predictions = logreg.predict(X_test_scaled)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print("Logistic Regression scaled Accuracy:", logreg_accuracy)

# Random Forest
rf = model2
rf.fit(X_train_scaled, y_train)
rf_predictions = rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)

# SVM
svm = model3
svm.fit(X_train_scaled, y_train)
svm_predictions = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# K-Nearest Neighbors (KNN)
knn = model4
knn.fit(X_train_scaled, y_train)
knn_predictions = knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("KNN Accuracy:", knn_accuracy)"""
########################################
"""

#cross validation of models for all age data 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform standard scaling on the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
models = [
    ('Logistic Regression', model1),
    ('Random Forest', model2),
    ('SVM', model3),
    ('KNN', model4),
    ('Decision Tree', model5)
]

# Evaluate and compare the models using cross-validation
for name, model in models:
    print(f"Evaluating {name}...")
    
    # Perform cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Print the cross-validation scores
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation score: {scores.mean()}\n")

"""



"""
from sklearn.preprocessing import MinMaxScaler




# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Min-Max scaling on the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
#svm = model3
svm = svm.SVC()
svm.fit(X_train_scaled, y_train)

# Make predictions on the test set
svm_predictions = svm.predict(X_test_scaled)

# Evaluate the model
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

"""

"""
#KFOLD

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold



# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}

# We will use a Support Vector Classifier with "rbf" kernel
svr = svm.SVC(kernel="rbf")

# Create inner and outer strategies
inner_cv = KFold(n_splits=2, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Pass the gridSearch estimator to cross_val_score
clf = GridSearchCV(estimator=svr, param_grid=p_grid, cv=inner_cv)
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv).mean()
print(nested_score)"""


 

 

