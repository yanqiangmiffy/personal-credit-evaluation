# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: 01_xgboost.py 
@Time: 2018/12/11 11:17
@Software: PyCharm 
@Description:
"""
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 加载数据
data = pd.read_csv('data/credit_pri_data.csv')
data['CFCS'] = data['CFCS'].fillna(0).astype('int32')  # 类别标签1
data['CFCS_Label'] = data['CFCS'].apply(lambda x: 1 if x != 0 else 0)  # 类别标签1
data.drop(columns=['JFCS','HOU_ID','PROPERTY_SIGN_DATE','PROPERTY_RECORD_DATE','Mortgage_starttime','Mortgage_endtime'],inplace=True,axis=1)
cate_cols = ['SEX', 'TEL_ID', 'PROVINCE', 'NATIONALITY', 'PROPERTY_ID', 'PROPERTY_USAGE_TYPE', 'PROPERTY_LOAN_WAY',
             'PROPERTY_PAYMENT']
data = pd.get_dummies(data, columns=cate_cols)

cols = [col for col in data.columns if col not in ['USER_ID', 'CFCS_Label', 'CFCS']]
X_train, X_test, y_train, y_test = train_test_split(data[cols], data['CFCS_Label'])

# 建立模型
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns)
dval = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns)
watchlist = [(dtrain, 'train'), (dval, 'val')]
print(X_test.columns)
params = {
    'eval_metric':['rmse','auc'],
    'objective': 'binary:logistic',
    'silent': True,
    'eta': 0.1,
    'max_depth': 6,
    'gamma': 10,
    'subsample': 0.95,
    'colsample_bytree': 1,
    'min_child_weight': 9,
    'scale_pos_weight': 1.2,
    'lambda': 250,
    # 'nthread': 15,
}
clf = xgb.train(
    params, dtrain,
    num_boost_round=100,
    evals=watchlist,
    verbose_eval=True
)

y_pred=clf.predict(xgb.DMatrix(X_test,feature_names=X_test.columns))
print(roc_auc_score(y_test,y_pred))

from xgboost import plot_importance
from matplotlib import pyplot

plot_importance(clf,max_num_features=20)
pyplot.show()