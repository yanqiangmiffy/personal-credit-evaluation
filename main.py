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
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

data = pd.read_csv('data/credit_pri_data.csv')


def process(df):
    """
    数据预处理
    :return:
    """
    # 使用平均值填充AGE的缺失值
    df['AGE'] = df['AGE'].fillna(df.mean())
    # 使用0 填充Sale_houcs/JFCS和CFCF的缺失值
    df['Sale_houcs'] = df['Sale_houcs'].fillna(0).astype('int32')
    df['JFCS'] = df['JFCS'].fillna(0).astype('int32')
    df['CFCS'] = df['CFCS'].fillna(0).astype('int32')  # 类别标签1
    df['CFCS_Label'] = df['CFCS'].apply(lambda x: 1 if x != 0 else 0)  # 类别标签1

    # 去除噪音的列或者无关紧要的列
    df.drop(
        columns=['SEX', 'HOU_ID', 'PROPERTY_SIGN_DATE', 'PROPERTY_RECORD_DATE', 'Mortgage_starttime',
                 'Mortgage_endtime'],
        inplace=True, axis=1)

    # 处理类别标签 one-hot
    cate_cols = ['TEL_ID', 'PROVINCE', 'NATIONALITY', 'PROPERTY_ID', 'PROPERTY_USAGE_TYPE', 'PROPERTY_LOAN_WAY',
                 'PROPERTY_PAYMENT']
    df = pd.get_dummies(df, columns=cate_cols)

    # 提取X和Y标签
    cols = [col for col in df.columns if col not in ['USER_ID', 'CFCS_Label', 'CFCS']]
    X = df[cols]
    y = df['CFCS_Label']
    print(df.shape[0] - df.count())

    return X, y


X, y = process(df=data)


def train_by_xgb():
    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_sample(X, y)
    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res)
    clf = XGBClassifier(max_depth=6)
    clf.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric='logloss',
            verbose=True)

    y_pred = clf.predict(x_test)
    label = pd.DataFrame()
    label['y_pred'] = y_pred
    print(label['y_pred'].value_counts())
    y_prob = clf.predict_proba(x_test)
    print(accuracy_score(y_test, y_pred))

    from xgboost import plot_importance
    from matplotlib import pyplot

    plot_importance(clf, max_num_features=20)
    pyplot.show()


train_test_split()
