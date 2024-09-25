import pandas as pd
import numpy as np

train = pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
X_train = train.drop(["outliers",'price'],axis=1)
y_train = train['price']
X_test = test.drop(["outliers",'price'],axis=1)
y_test = test['price']


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error



n_clusters = 3


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),index=X_test.index, columns=X_test.columns)




from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(X_train)
X_train['cluster'] = clusters

models = {}
for cluster in range(n_clusters):
    data = pd.concat([X_train,y_train],axis=1)
    cluster_data = data[data['cluster'] == cluster]
    X_cluster = cluster_data.drop(['price', 'cluster'], axis=1)
    y_cluster = cluster_data['price']
    model = RandomForestRegressor()
    model.fit(X_cluster,y_cluster)
    models[cluster] = model


test_clusters = kmeans.predict(X_test)

pred = []
for i, cluster in enumerate(test_clusters):
    model = models[cluster]
    terefere = model.predict([X_test.iloc[i]])
    pred.append(terefere[0])


mse = mean_squared_error(pred,y_test)
print("MSE:",mse)
sq_mean = np.sqrt(mse/len(pred))
print(sq_mean)
