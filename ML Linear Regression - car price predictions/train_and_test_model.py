import pandas as pd
import numpy as np
from scipy import stats

train = pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
X_train = train.drop(["outliers",'price'],axis=1)
y_train = train['price']
X_test = test.drop(["outliers",'price'],axis=1)
y_test = test['price']


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(1,11))
X_train = pd.DataFrame(scaler.fit_transform(X_train),index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),index=X_test.index, columns=X_test.columns)

for col in X_train.columns:
    X_train[col], _ = stats.boxcox(X_train[col])
    print(col,'-done')

for col in X_test.columns:
    X_test[col], _ = stats.boxcox(X_test[col])
    print(col, '-done')



from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
clf = RandomForestRegressor()
#clf = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.01)

clf.fit(X_train,y_train)
pred = clf.predict(X_test)
mse = mean_squared_error(pred,y_test)
print("MSE:",mse)
sq_mean = np.sqrt(mse/len(pred))
print(sq_mean)

