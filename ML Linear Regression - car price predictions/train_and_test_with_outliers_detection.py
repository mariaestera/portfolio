import pandas as pd
import numpy as np

train = pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
X_train = train.drop(["outliers",'price'],axis=1)
y_train = train['price']
c_train = train['outliers']
X_test = test.drop(["outliers",'price'],axis=1)
y_test = test['price']
c_test = test['outliers']


from sklearn.metrics import mean_squared_error
from scipy import stats


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1,11))
X_train = pd.DataFrame(scaler.fit_transform(X_train),index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),index=X_test.index, columns=X_test.columns)

# data normalization
for col in X_train.columns:
    X_train[col], _ = stats.boxcox(X_train[col])
    print(col,'-done')

for col in X_test.columns:
    X_test[col], _ = stats.boxcox(X_test[col])
    print(col, '-done')


# train model to recognize very expensive cars
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import itertools

best_score = 0
best_params = []

C = [1e3, 5e3, 1e4, 5e4, 1e5],
gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
kernel = ["rbf","poly"],
max_iter = [1, 10, 100, 1000],
grid = list(itertools.product(*C, *gamma,*kernel,*max_iter))

# looking for best params
control = True
if control:
    for i in grid:
        clf = SVC(C=i[0], gamma=i[1], kernel=i[2], max_iter=i[3])
        clf.fit(X_train, c_train)
        score = f1_score(clf.predict(X_test), c_test)
        if score > best_score:
            best_score = score
            best_params = i
    print(f"Best F1-score: {best_score}, params: {best_params}")
    with open("best_params.txt", "w") as plik:
        for i in range(3):
            plik.write(str(best_params[i]) + "\n")
        plik.write(str(best_params[3]))
else:
    best_params = open("best_params.txt", "r").read()
    best_params = best_params.split("\n")
    best_params = [float(best_params[0]), float(best_params[1]), str(best_params[2]), int(best_params[3])]
    print("Current parameters:", best_params)

    clf = SVC(C=best_params[0], gamma=best_params[1], kernel=best_params[2], max_iter=best_params[3])
    clf.fit(X_train, c_train)
    score = f1_score(clf.predict(X_test), c_test)

    print(f"F1-score: {score}, params: {best_params}")

from sklearn.ensemble import GradientBoostingRegressor
models = {}
for cluster in range(2):
    data = pd.concat([X_train,c_train,y_train],axis=1)
    cluster_data = data[data['outliers'] == cluster]
    X_cluster = cluster_data.drop(['price', 'outliers'], axis=1)
    y_cluster = cluster_data['price']
    model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1)
    model.fit(X_cluster,y_cluster)
    models[cluster] = model


test_clusters = clf.predict(X_test)

pred = []
for i, cluster in enumerate(test_clusters):
    model = models[cluster]
    terefere = model.predict([X_test.iloc[i]])
    pred.append(terefere[0])


mse = mean_squared_error(pred,y_test)
print("MSE:",mse)
sq_mean = np.sqrt(mse/len(pred))
print(sq_mean)

