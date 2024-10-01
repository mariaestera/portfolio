import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()
from functions import make_mi_scores, visualize_barplot, show_frame
show_visualization = False

data = pd.read_csv("train.csv",index_col=0)

#how much data from every cathegory we have?
print(data.groupby('NObeyesdad').Gender.count())
#classes are relatively balanced

#check more about how ouer database looks like
print(data.groupby('Gender').NObeyesdad.count())
# it's ok - we have almost 50-50 man and women in database

#replace some simply discrete features by numbers
data['Gender']= data['Gender'].replace({'Male':0, 'Female':1})
data['family_history_with_overweight'] = data['family_history_with_overweight'].replace({'yes':1, 'no':0})
data['SMOKE']= data['SMOKE'].replace({'no':0, 'yes':1})
data['FAVC']= data['FAVC'].replace({'no':0, 'yes':1})
data['SCC']= data['SCC'].replace({'no':0, 'yes':1})
data['CAEC']=data['CAEC'].replace({'Sometimes':1, 'Frequently':2, 'no':0, 'Always':3})
data['CALC']=data['CALC'].replace({'Sometimes':1, 'Frequently':2, 'no':0})
#ouer target:
data['NObeyesdad'] = data['NObeyesdad'].replace({'Insufficient_Weight':0,'Normal_Weight':1,'Overweight_Level_I':2,'Overweight_Level_II':3,'Obesity_Type_I':4,'Obesity_Type_II':5,'Obesity_Type_III':6})

# checking results
if show_visualization:
    visualize_barplot('CAEC',data)
    visualize_barplot('CALC',data)
#we don't have linear relationship!


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop('NObeyesdad',axis=1),data.NObeyesdad,test_size=0.2, random_state=42)

from category_encoders import MEstimateEncoder
encoder = MEstimateEncoder(cols=['MTRANS','CAEC','CALC'],m=20.0)
encoder.fit(X_train,y_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)


if show_visualization:
    visualize_barplot('CAEC',pd.concat([X_train,y_train],axis=1))
    visualize_barplot('CALC',pd.concat([X_train,y_train],axis=1))


#creating features
for i in [X_train,X_test]:
    i['BMI'] = i['Weight']/(i['Height']*i['Height'])

mi_scores = make_mi_scores(X_train,y_train)
print(mi_scores)

# drop "SMOKE" - it have the worst MI score and this drop improves model scoring
X_train = X_train.drop(['SMOKE'],axis=1)
X_test = X_test.drop(['SMOKE'],axis = 1)

#let's find features with linear correlation with target
correlation = pd.Series()
for i in X_train.columns:
    new = pd.Series({i: y_train.corr(X_train[i])})
    correlation = pd.concat([correlation, new])
print(correlation.sort_values())
#SCC had either bad mi-score and correlation with target
X_train = X_train.drop(['SCC'],axis=1)
X_test = X_test.drop(['SCC'],axis = 1)


#how features are related to each other?
correlation_matrix = X_train.corr()
correlation_matrix = correlation_matrix.round(3)
if show_visualization:
    show_frame(correlation_matrix)


# scaling and normalizing data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_train = pd.DataFrame(X_train_s,index=X_train.index,columns=X_train.columns)
X_test_s = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_s,index=X_test.index,columns=X_test.columns)


# train and test classifiers
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

clf1 = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [100,1000],
    'learning_rate': [0.1,0.001,0.0001]
}

grid_search = GridSearchCV(clf1, param_grid, cv=5)
grid_search.fit(X_train, y_train)
clf1 = grid_search.best_estimator_


clf2 = SVC()

param_grid = {
    'C': [0.001, 0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(clf2, param_grid, cv=5)
grid_search.fit(X_train, y_train)
clf2 = grid_search.best_estimator_

clf = VotingClassifier([('gnb_1',clf1),('svc',clf2)])
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
