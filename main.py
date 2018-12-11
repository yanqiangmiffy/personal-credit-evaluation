# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: main.py 
@Time: 2018/12/5 11:43
@Software: PyCharm 
@Description:
"""
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
data=pd.read_csv('data/credit_pri_data.csv')
data['CFCS_Label']=data['CFCS'].apply(lambda x:1 if x!=0 else 0) # 类别标签1

# print(data[data['CFCS']!=0][['CFCS','CFCS_Label','JFCS']])
# print(data.info())
# print(data.USER_ID.value_counts())
data.drop(columns=['SEX','HOU_ID','PROPERTY_SIGN_DATE','PROPERTY_RECORD_DATE','Mortgage_starttime','Mortgage_endtime'],inplace=True,axis=1)

cate_cols=['TEL_ID','PROVINCE','NATIONALITY','PROPERTY_ID','PROPERTY_USAGE_TYPE','PROPERTY_LOAN_WAY','PROPERTY_PAYMENT']
data=pd.get_dummies(data,columns=cate_cols)
print(data.columns)
print(data.shape)

cols=[col for col in data.columns if col not in ['USER_ID','CFCS_Label','CFCS']]
clf=XGBClassifier(max_depth=6)
X_train,X_test,y_train,y_test=train_test_split(data[cols],data['CFCS_Label'])
clf.fit(X_train,y_train,
        eval_set=[(X_test,y_test)],
        eval_metric='logloss',
        verbose=True)

y_pred=clf.predict(X_test)
label=pd.DataFrame()
label['y_pred']=y_pred
print(label['y_pred'].value_counts())
y_prob=clf.predict_proba(X_test)
print(y_prob)
print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


from sklearn.cluster import AffinityPropagation
clf=AffinityPropagation(preference=-50)
