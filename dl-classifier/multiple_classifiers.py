# Author: McKenzie Hawkins, mjh244
# Performs classification using a variety of ML algorithms

# Imports
import matplotlib.pyplot as plt
import csv
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Method to compute percent of true positives, false positives, etc.
def measure(actual, predictions):
  tpr = 0
  tnr = 0
  tpSum = 0
  fpSum = 0
  tnSum = 0
  fnSum = 0
  for i in range(len(actual)):
    if ((actual[i] == 1) and (predictions[i] == 1)):
      tpSum += 1
    elif ((actual[i] == 0) and (predictions[i] == 1)):
      fpSum += 1
    elif ((actual[i] == 0) and (predictions[i] == 0)):
      tnSum += 1
    elif ((actual[i] == 1) and (predictions[i] == 0)):
      fnSum += 1 
  tpr = tpSum / (tpSum + fnSum)
  tnr = tnSum / (tnSum + fpSum)
  return tpr, tnr
  


# Lists to hold data
data = []
features = []
labels = []

# Reads the features and labels from the files (have to uncomment and rerun for different files)
with open('/biomarker-id/data/brca-normalized.csv','r') as file: 
#with open('/biomarker-id/data/PIK3CA-normalized.csv','r') as file: 
#with open('/biomarker-id/data/TP53-normalized.csv','r') as file: 
#with open('/biomarker-id/data/CDH1-normalized.csv','r') as file: 
  for line in csv.reader(file):
    if (line[0] != "0"):
      data.append(line) 
      features.append(line[0:24])
      labels.append(line[24])

# Prints out features and labels for verification
print("Features")
print(features)
print()
print("Labels")
print(labels)
print()

# Initializes alive and died lists to record patient outcomes
alive = 0
died = 0

# Converts elements in labels and features lists to float
for i in range(len(labels)):
  labels[i] = float(labels[i])
  if (labels[i] == 0.0):
    died += 1
  else:
    alive += 1
for i in range(len(features)):
  for j in range(len(features[i])):
    features[i][j] = float(features[i][j])

# Prints number of alive vs dead and probs from dataset
print("Alive: ", alive)
print("Died: ", died)
print("Probability of living: ", alive / (alive + died))
print("Probability of dying: ", died / (alive + died))


# Initalizes variables to hold accuracy sum
dt_acc = 0
svc_acc = 0
linearSVC_acc = 0
sgd_acc = 0
gnb_acc = 0
compNB_acc = 0
lr_acc = 0
kNN_acc = 0
rf_acc = 0

# Initalizes variables to hold true positive and negative rates
dt_ttr, dt_tnr = 0, 0
svc_ttr, svc_tnr = 0, 0
linearSVC_ttr, linearSVC_tnr = 0, 0
sgd_ttr, sgd_tnr = 0, 0
gnb_ttr, gnb_tnr = 0, 0
compNB_ttr, compNB_tnr = 0, 0
lr_ttr, lr_tnr = 0, 0
kNN_ttr, kNN_tnr = 0, 0
rf_ttr, rf_tnr = 0, 0

# Trains/tests the models 100 times and sums results to average at the end
for i in range(100):

  # Splits data into 80% training and 20% testing
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

  # Creates the decision tree
  classifierTree = tree.DecisionTreeClassifier(max_depth=7, random_state=3)
  # Fits the tree to the data
  classifierTree = classifierTree.fit(features_train, labels_train)
  # Makes the predictions
  dt_predictions = classifierTree.predict(features_test)
  # Adds to running total of accuracy scores
  dt_acc += accuracy_score(labels_test, dt_predictions)
  # Adds to running total of true positive and true negative rate
  dt_ttr += measure(labels_test, dt_predictions)[0]
  dt_tnr += measure(labels_test, dt_predictions)[1]

  # SVC model
  classifierSVC = svm.SVC(kernel = 'linear', class_weight='balanced')
  classifierSVC = classifierSVC.fit(features_train, labels_train)
  svc_predictions = classifierSVC.predict(features_test)
  svc_acc += accuracy_score(labels_test, svc_predictions)
  svc_ttr += measure(labels_test, svc_predictions)[0]
  svc_tnr += measure(labels_test, svc_predictions)[1]

  # Linear SVM model
  classifierSVM = svm.LinearSVC(class_weight='balanced')
  classifierSVM = classifierSVM.fit(features_train, labels_train)
  linearSVC_predictions = classifierSVM.predict(features_test)
  linearSVC_acc += accuracy_score(labels_test, linearSVC_predictions)
  linearSVC_ttr += measure(labels_test, linearSVC_predictions)[0]
  linearSVC_tnr += measure(labels_test, linearSVC_predictions)[1]

  # Stochastic Gradient Descent model
  classifierSGD = SGDClassifier()
  classifierSGD = classifierSGD.fit(features_train, labels_train)
  sgd_predictions = classifierSGD.predict(features_test)
  sgd_acc += accuracy_score(labels_test, sgd_predictions)
  sgd_ttr += measure(labels_test, sgd_predictions)[0]
  sgd_tnr += measure(labels_test, sgd_predictions)[1]

  # Gaussian Naive Bayes model
  classifierNB = GaussianNB()
  classifierNB = classifierNB.fit(features_train, labels_train)
  gnb_predictions = classifierNB.predict(features_test)
  gnb_acc += accuracy_score(labels_test, gnb_predictions)
  gnb_ttr += measure(labels_test, gnb_predictions)[0]
  gnb_tnr += measure(labels_test, gnb_predictions)[1]

  # Complement Naive Bayes model
  classifierNB = ComplementNB()
  classifierNB = classifierNB.fit(features_train, labels_train)
  compNB_predictions = classifierNB.predict(features_test)
  compNB_acc += accuracy_score(labels_test, compNB_predictions)
  compNB_ttr += measure(labels_test, compNB_predictions)[0]
  compNB_tnr += measure(labels_test, compNB_predictions)[1]

  # Logistic Regression model
  classifierLR = LogisticRegression()
  classifierLR = classifierLR.fit(features_train, labels_train)
  lr_predictions = classifierLR.predict(features_test)
  lr_acc += accuracy_score(labels_test, lr_predictions)
  lr_ttr += measure(labels_test, lr_predictions)[0]
  lr_tnr += measure(labels_test, lr_predictions)[1]

  # K Nearest Neighbours model
  classifierKNN = KNeighborsClassifier()
  classifierKNN = classifierKNN.fit(features_train, labels_train)
  kNN_predictions = classifierKNN.predict(features_test)
  kNN_acc += accuracy_score(labels_test, kNN_predictions)
  kNN_ttr += measure(labels_test, kNN_predictions)[0]
  kNN_tnr += measure(labels_test, kNN_predictions)[1]

  # Random Forest model
  classifierRF = RandomForestClassifier()
  classifierRF = classifierRF.fit(features_train, labels_train)
  rf_predictions = classifierRF.predict(features_test)
  rf_acc += accuracy_score(labels_test, rf_predictions)
  rf_ttr += measure(labels_test, rf_predictions)[0]
  rf_tnr += measure(labels_test, rf_predictions)[1]

# Averages the accuracy scores
dt_acc = dt_acc / 100
svc_acc = svc_acc / 100
linearSVC_acc = linearSVC_acc / 100
sgd_acc = sgd_acc / 100
gnb_acc = gnb_acc / 100
compNB_acc = compNB_acc / 100
lr_acc = lr_acc / 100
kNN_acc = kNN_acc / 100
rf_acc = rf_acc / 100

# Averages the true positive and true nefative rates
dt_ttr = dt_ttr / 100
dt_tnr = dt_tnr / 100
svc_ttr = svc_ttr / 100
svc_tnr = svc_tnr / 100
linearSVC_ttr = linearSVC_ttr / 100
linearSVC_tnr = linearSVC_tnr / 100
sgd_ttr = sgd_ttr / 100
sgd_tnr = sgd_tnr / 100
gnb_ttr = gnb_ttr / 100
gnb_tnr = gnb_tnr / 100
compNB_ttr = compNB_ttr / 100
compNB_tnr = compNB_tnr / 100
lr_ttr = lr_ttr / 100
lr_tnr = lr_tnr / 100
kNN_ttr = kNN_ttr / 100
kNN_tnr = kNN_tnr / 100
rf_ttr = rf_ttr / 100
rf_tnr = rf_tnr / 100

# Plots the decision tree image
#plt.figure(figsize=(10,8))
#tree.plot_tree(classifierTree) 

# Prints out relevant scores
print("Accuracy score of decision tree:", dt_acc)
print("TPR and TNR of decision tree:", dt_ttr, ", ", dt_tnr)
print("Accuracy score of SVC:", svc_acc)
print("TPR and TNR of SVC:", svc_ttr, ", ", svc_tnr)
print("Accuracy score of Linear SVC:", linearSVC_acc)
print("TPR and TNR of Linear SVC:", linearSVC_ttr, ", ", linearSVC_tnr)
print("Accuracy score of SGD:", sgd_acc)
print("TPR and TNR of SGD:", sgd_ttr, ", ", sgd_tnr)
print("Accuracy score of Gaussian NB:", gnb_acc)
print("TPR and TNR of Gaussian NB:", gnb_ttr, ", ", gnb_tnr)
print("Accuracy score of Complementary NB:", compNB_acc)
print("TPR and TNR of Complementary NB:", compNB_ttr, ", ", compNB_tnr)
print("Accuracy score of LR:", lr_acc)
print("TPR and TNR of LR:", lr_ttr, ", ", lr_tnr)
print("Accuracy score of KNN:", kNN_acc)
print("TPR and TNR of KNN:", kNN_ttr, ", ", kNN_tnr)
print("Accuracy score of RF:", rf_acc)
print("TPR and TNR of RF:", rf_ttr, ", ", rf_tnr)