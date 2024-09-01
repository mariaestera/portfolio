import pandas as pd

#import data and extract labels

all_data = pd.read_csv("train.csv",index_col=0)
print('Data imported')
print(all_data.columns)
labels = all_data['class'].replace({'e': 0, 'p': 1})
features = all_data.drop(['class'],axis=1)


#divide data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_valid,y_train,y_valid = train_test_split(features,labels,test_size=0.2)



# find categorical and numerical data
ob_cols1 = [col for col in X_train.columns if pd.api.types.is_object_dtype(X_train[col]) and X_train[col].nunique()<=5]
ob_cols2 = [col for col in X_train.columns if pd.api.types.is_object_dtype(X_train[col]) and X_train[col].nunique()>5]
num_cols = list(set(X_train.columns) - (set(ob_cols1)|set(ob_cols2)))

#replace missing values
from sklearn.impute import SimpleImputer
imputer_num = SimpleImputer(strategy='median')
imputer_ob = SimpleImputer(strategy='most_frequent')

X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
X_train[ob_cols1+ob_cols2] = imputer_ob.fit_transform(X_train[ob_cols1+ob_cols2])
X_valid[num_cols] = imputer_num.fit_transform(X_valid[num_cols])
X_valid[ob_cols1+ob_cols2] = imputer_ob.fit_transform(X_valid[ob_cols1+ob_cols2])


#dealing with non-numerical values

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
OEH = OneHotEncoder(min_frequency=0.01,sparse_output=False)
OE = OrdinalEncoder()


ob_train1 = pd.DataFrame(OEH.fit_transform(X_train[ob_cols1]))
ob_train2 = pd.DataFrame(OE.fit_transform(X_train[ob_cols2]))
ob_train1.index = X_train.index
ob_train2.index = X_train.index

ob_valid1 = pd.DataFrame(OEH.transform(X_valid[ob_cols1]))
ob_valid2 = pd.DataFrame(OE.transform(X_valid[ob_cols2]))
ob_valid1.index = X_valid.index
ob_valid2.index = X_valid.index


#joining data num-ob1-ob2
X_train = pd.concat([X_train.drop(ob_cols1+ob_cols2,axis=1),ob_train1,ob_train2],axis=1)
X_valid = pd.concat([X_valid.drop(ob_cols1+ob_cols2,axis=1),ob_valid1,ob_valid2],axis=1)
X_train.columns = X_train.columns.astype(str)
X_valid.columns = X_valid.columns.astype(str)



from sklearn.ensemble import RandomForestClassifier
my_model = RandomForestClassifier()

my_model.fit(X_train,y_train)
print(my_model.score(X_valid,y_valid))
