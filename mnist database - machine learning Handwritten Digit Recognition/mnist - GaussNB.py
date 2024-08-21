from sklearn.datasets import fetch_openml
from time import time
import numpy as np
from function import sample_results

# downloading data
mnist = fetch_openml('mnist_784', version=1)

images, labels = mnist['data'], mnist['target']
labels = labels.astype(int)

# show first image
import matplotlib.pyplot as plt
first_image = images.iloc[0].values.reshape(28, 28)
plt.imshow(first_image, cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.show(block=False)

# train-test split
from sklearn.model_selection import train_test_split, GridSearchCV
images_train_0, images_test_0, labels_train, labels_test = train_test_split(images, labels, test_size=0.25, random_state=42)

# PCA
n=100 #number of dimensions (start point - 728)
from sklearn.decomposition import PCA
pca = PCA(n_components=n)
images_train = pca.fit_transform(images_train_0)
images_test = pca.transform(images_test_0)
print("PCA - done")

from sklearn.naive_bayes import GaussianNB
print("Fitting GaussianNB classifier to the training set")
t0 = time()
clf = GaussianNB()
clf = clf.fit(images_train, labels_train)
print("done in %0.3fs" % (time() - t0))
print("Accuracy: ",clf.score(images_test,labels_test))


from sklearn.metrics import classification_report

Gauss_pred = clf.predict(images_test)
print(classification_report(labels_test, Gauss_pred, target_names=[str(i) for i in range(10)]))

# sample results
sample_results(images_test_0,Gauss_pred,k=30)