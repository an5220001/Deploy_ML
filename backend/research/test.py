import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import joblib


features = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race" , "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
data_train = pd.read_csv("adult.data", sep=",", names=features)
# print(data_train.head())
X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
# print(X_train.head())
# print(y_train.head())
# print(X_train.shape, y_train.shape)

data_text = pd.read_csv("adult.test", sep=",", names=features)
X_test = data_text.iloc[:, :-1]
y_test = data_text.iloc[:, -1]
# print(X_test.shape, y_test.shape)

train_mode = dict(X_train.mode().iloc[0]) # mode() để lấy giá trị xuất hiện nhiều và iloc[0] để lấy chọn giá trị đầu tiên
X_train = X_train.fillna(train_mode) # fill giá trị NaN, Na, Null, ... bằng giá trị mới tìm được (mode)

features_categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country']
encoder = {}
for col in features_categorical:
    X_train[col] = LabelEncoder().fit_transform(X_train[col])
    encoder[col] = LabelEncoder()
# print(X_train.head())

for col in features[:-1]:
    if X_train[col].dtypes != "int64":
        X_train[col] = X_train[col].astype('int64')

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(X_train, y_train)

et = ExtraTreesClassifier(n_estimators=100)
et = et.fit(X_train, y_train)

joblib.dump(train_mode, "train_mode.joblib", compress=True)
joblib.dump(encoder, "encoder.joblib", compress=True)
joblib.dump(rf, "rf.joblib", compress=True)
joblib.dump(et, "et.joblib", compress=True)
