import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from time import time
iris = load_iris()

#what data we have? projection 3 of 4 dimensions
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(iris.data[:, 0], iris.data[:, 1],iris.data[:, 2],c=iris.target)
plt.show()

#PCA
from sklearn.decomposition import PCA
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
X = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X[:, 0],X[:, 1],X[:, 2],c=iris.target)
plt.show()



# preparation data
from sklearn.model_selection import train_test_split, GridSearchCV
data_train, data_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.50, random_state=32)
# PCA only for training set
pca = PCA(n_components=3)
X = pca.fit_transform(data_train)
data_test = pca.transform(data_test)

#SVC
from sklearn.svm import SVC

print("Fitting the classifier...")
t0 = time()
param_grid = {
          'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }

clf = GridSearchCV(SVC(), param_grid)
clf = clf.fit(X, labels_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

accuracy = clf.score(data_test, labels_test)
print("Accuracy of SVC:",accuracy)

# Gaussian Naive bayes
from sklearn.naive_bayes import GaussianNB

print("Fitting the classifier...")
t0 = time()
clf = GaussianNB()
clf = clf.fit(X, labels_train)
print("done in %0.3fs" % (time() - t0))

accuracy = clf.score(data_test, labels_test)
print("Accuracy of GaussianNB:",accuracy)

from sklearn.ensemble import RandomForestClassifier
print("Fitting the classifier...")
t0 = time()
param_grid = {
        'max_depth': [3,5,10],
        'min_samples_split':[2,5,10,100]
          }

clf = GridSearchCV(RandomForestClassifier(), param_grid)
clf = clf.fit(X, labels_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

accuracy = clf.score(data_test, labels_test)
print("Accuracy of RandomForrestClasiffier:",accuracy)

#K nearest neibourhods
from sklearn.neighbors import KNeighborsClassifier

print("Fitting the classifier...")
t0 = time()
param_grid = {
        'n_neighbors': [3,5,10]
          }

clf = GridSearchCV(KNeighborsClassifier(), param_grid)
clf = clf.fit(X, labels_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

accuracy = clf.score(data_test, labels_test)
print("Accuracy of KNeighborsClassifier:",accuracy)

