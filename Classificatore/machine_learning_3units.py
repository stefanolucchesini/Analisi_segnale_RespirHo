# -*- coding: utf-8 -*-
"""Machine Learning_3units.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kFlL0CTvy8OPFUt1vjYpl3WrhtYpcqbb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import metrics
import seaborn as sns
import io 
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from google.colab import files
uploaded = files.upload()
# caricamento dataset con le features
dataset = pd.read_csv(io.BytesIO(uploaded['featurestot.csv'])) 

dataset["activity"] = dataset["activity"].astype("|S")
#features nulle vengono cancellate
dataset = dataset.dropna()
#hot encoding
label_encoder = LabelEncoder()
dataset["activitynum"] = label_encoder.fit_transform(dataset["activity"])

y = dataset['activitynum']
X = dataset[['mean_quat0', 'mean_quat1','mean_quat2', 'mean_quat3','mean_quat4', 'mean_quat5', 'mean_quat6', 'mean_quat7', 'mean_quat8', 'mean_quat9', 'mean_quat10', 'mean_quat11', 'std_quat0', 'std_quat1', 'std_quat2', 'std_quat3', 'std_quat4', 'std_quat5', 'std_quat6', 'std_quat7', 'std_quat8', 'std_quat9', 'std_quat10', 'std_quat11',
             'skew_quat0', 'skew_quat1', 'skew_quat2', 'skew_quat3', 'skew_quat4', 
                                                                        'skew_quat5', 'skew_quat6', 'skew_quat7', 'skew_quat8', 'skew_quat9', 
                                                                        'skew_quat10', 'skew_quat11', 'rms_quat0', 'rms_quat1', 'rms_quat2', 'rms_quat3', 'rms_quat4', 'rms_quat5', 'rms_quat6', 'rms_quat7', 'rms_quat8', 'rms_quat9', 'rms_quat10', 'rms_quat11','kur_quat0', 'kur_quat1', 'kur_quat2', 'kur_quat3', 'kur_quat4', 'kur_quat5', 'kur_quat6', 'kur_quat7', 'kur_quat8', 'kur_quat9', 'kur_quat10', 'kur_quat11', 'ptp_quat0', 'ptp_quat1', 'ptp_quat2', 'ptp_quat3', 'ptp_quat4', 'ptp_quat5', 'ptp_quat6', 'ptp_quat7', 'ptp_quat8', 'ptp_quat9', 'ptp_quat10', 'ptp_quat11', 'iqr_quat0', 'iqr_quat1', 'iqr_quat2', 'iqr_quat3', 'iqr_quat4', 'iqr_quat5', 'iqr_quat6', 'iqr_quat7', 'iqr_quat8', 'iqr_quat9', 'iqr_quat10', 'iqr_quat11',
             'median_quat0', 'median_quat1', 'median_quat2', 'median_quat3', 'median_quat4', 'median_quat5', 'median_quat6', 'median_quat7', 'median_quat8', 'median_quat9', 'median_quat10', 'median_quat11','crestf_quat0', 'crestf_quat1', 'crestf_quat2', 'crestf_quat3', 'crestf_quat4', 'crestf_quat5', 'crestf_quat6', 'crestf_quat7', 'crestf_quat8', 'crestf_quat9', 'crestf_quat10', 'crestf_quat11'
        ]]

# standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
dataset = pd.DataFrame(data = X, columns = ['mean_quat0', 'mean_quat1','mean_quat2', 'mean_quat3','mean_quat4', 'mean_quat5', 'mean_quat6', 'mean_quat7', 'mean_quat8', 'mean_quat9', 'mean_quat10', 'mean_quat11', 'std_quat0', 'std_quat1', 'std_quat2', 'std_quat3', 'std_quat4', 'std_quat5', 'std_quat6', 'std_quat7', 'std_quat8', 'std_quat9', 'std_quat10', 'std_quat11',
             'skew_quat0', 'skew_quat1', 'skew_quat2', 'skew_quat3', 'skew_quat4', 
                                                                        'skew_quat5', 'skew_quat6', 'skew_quat7', 'skew_quat8', 'skew_quat9', 
                                                                        'skew_quat10', 'skew_quat11', 'rms_quat0', 'rms_quat1', 'rms_quat2', 'rms_quat3', 'rms_quat4', 'rms_quat5', 'rms_quat6', 'rms_quat7', 'rms_quat8', 'rms_quat9', 'rms_quat10', 'rms_quat11','kur_quat0', 'kur_quat1', 'kur_quat2', 'kur_quat3', 'kur_quat4', 'kur_quat5', 'kur_quat6', 'kur_quat7', 'kur_quat8', 'kur_quat9', 'kur_quat10', 'kur_quat11', 'ptp_quat0', 'ptp_quat1', 'ptp_quat2', 'ptp_quat3', 'ptp_quat4', 'ptp_quat5', 'ptp_quat6', 'ptp_quat7', 'ptp_quat8', 'ptp_quat9', 'ptp_quat10', 'ptp_quat11', 'iqr_quat0', 'iqr_quat1', 'iqr_quat2', 'iqr_quat3', 'iqr_quat4', 'iqr_quat5', 'iqr_quat6', 'iqr_quat7', 'iqr_quat8', 'iqr_quat9', 'iqr_quat10', 'iqr_quat11',
             'median_quat0', 'median_quat1', 'median_quat2', 'median_quat3', 'median_quat4', 'median_quat5', 'median_quat6', 'median_quat7', 'median_quat8', 'median_quat9', 'median_quat10', 'median_quat11','crestf_quat0', 'crestf_quat1', 'crestf_quat2', 'crestf_quat3', 'crestf_quat4', 'crestf_quat5', 'crestf_quat6', 'crestf_quat7', 'crestf_quat8', 'crestf_quat9', 'crestf_quat10', 'crestf_quat11'])
dataset['activity'] = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42, 
                                                    stratify = y)

#pca
pca = PCA(n_components=2)
pca.fit(X_train)
pca.fit(X_test)

#tutti i modelli con le relative cm e report
"""# KNN"""

model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

labels = ['cyclette', 'lying_left', 'lying_right', 'prone', 'running', 'sitting',
          #'sitting_with_support','sitting_without_support', 
          'stairs', 'standing', 'supine', 
          'walking']

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
def print_confusionMatrix(y_test, rounded_y_pred):
  cm = confusion_matrix(y_test, rounded_y_pred)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16,7))
sns.heatmap(cm, cmap = "Blues", annot = True, fmt = ".1f", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()
    
print("-"*125)

precision = cm/cm.sum(axis = 0)
sns.set(font_scale=1.5)

plt.figure(figsize=(16,7))
sns.heatmap(precision, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
plt.title("Precision Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()
    
print("-"*125)

recall = (cm.T/cm.sum(axis = 1)).T

plt.figure(figsize=(16,7))
sns.heatmap(recall, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
plt.title("Recall Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()

scores = []
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors = i, n_jobs = -1)
    knn.fit(X_train, y_train.ravel())
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy Score')
xticks = range(1,50)
plt.plot(xticks, scores, color='red', linestyle='solid', marker='o',
         markerfacecolor='blue', markersize=5)
plt.show()

"""# Random Forest Classifier"""

classifier = RandomForestClassifier(n_jobs = 1,random_state =0)
classifier.fit(X_train,y_train.ravel())
y_pred=classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

labels = ['cyclette', 'lying_left', 'lying_right', 'prone', 'running', 'sitting',
          #'sitting_with_support','sitting_without_support', 
          'stairs', 'standing', 'supine', 
          'walking']
          #'walking_fast', 'walkig_slow']
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
def print_confusionMatrix(y_test, rounded_y_pred):
  cm = confusion_matrix(y_test, rounded_y_pred)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16,7))
sns.heatmap(cm, cmap = "Blues", annot = True, fmt = ".1f", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()
    
print("-"*125)

precision = cm/cm.sum(axis = 0)
sns.set(font_scale=1.5)

plt.figure(figsize=(16,7))
sns.heatmap(precision, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
plt.title("Precision Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()
    
print("-"*125)

recall = (cm.T/cm.sum(axis = 1)).T

plt.figure(figsize=(16,7))
sns.heatmap(recall, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
plt.title("Recall Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()

"""#SVC"""

classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

labels = ['cyclette', 'lying_left', 'lying_right', 'prone', 'running', 
          #'sitting_with_support','sitting_without_support', 
          'sitting',
          'stairs', 'standing', 'supine', 
          'walking']
          #'walking_fast', 'walkig_slow']
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
def print_confusionMatrix(y_test, rounded_y_pred):
  cm = confusion_matrix(y_test, rounded_y_pred)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16,7))
sns.heatmap(cm, cmap = "Blues", annot = True, fmt = ".1f", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()
    
print("-"*125)

precision = cm/cm.sum(axis = 0)
sns.set(font_scale=1.5)

plt.figure(figsize=(16,7))
sns.heatmap(precision, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
plt.title("Precision Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()
    
print("-"*125)

recall = (cm.T/cm.sum(axis = 1)).T

plt.figure(figsize=(16,7))
sns.heatmap(recall, cmap = "Blues", annot = True, fmt = ".2f", xticklabels=labels, yticklabels=labels)
plt.title("Recall Matrix", fontsize = 30)
plt.xlabel('Predicted Class', fontsize = 20)
plt.ylabel('Original Class', fontsize = 20)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 90)
plt.show()

!pip install tpot

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv(io.BytesIO(uploaded['ref_complete.txt'])) 
dataset.head()

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,[-1]].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline.py')