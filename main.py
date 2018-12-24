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
import matplotlib.pyplot as plt

def process_for_classification(df):
    """
    分类模型 数据预处理
    :return:
    """
    print("proces raw data...")
    # 使用平均值填充AGE的缺失值
    df['AGE'] = df['AGE'].fillna(df['AGE'].mean())
    # 使用0 填充Sale_houcs/JFCS和CFCF的缺失值
    df['Purchase_houcs'] = df['Purchase_houcs'].fillna(0).astype('int32')
    df['Sale_houcs'] = df['Sale_houcs'].fillna(0).astype('int32')
    df['JFCS'] = df['JFCS'].fillna(0).astype('int32')
    df['CFCS'] = df['CFCS'].fillna(0).astype('int32')  # 类别标签1
    df['CFCS_Label'] = df['CFCS'].apply(lambda x: 1 if x != 0 else 0)  # 类别标签1

    # 去除噪音的列或者无关紧要的列
    df.drop(
        columns=['SEX', 'HOU_ID', 'PROPERTY_SIGN_DATE', 'PROPERTY_RECORD_DATE', 'Mortgage_starttime',
                 'Mortgage_endtime', 'JFCS'],
        inplace=True, axis=1)

    # 处理类别标签 one-hot
    cate_cols = ['TEL_ID', 'PROVINCE', 'NATIONALITY', 'PROPERTY_ID', 'PROPERTY_USAGE_TYPE', 'PROPERTY_LOAN_WAY',
                 'PROPERTY_PAYMENT']
    df = pd.get_dummies(df, columns=cate_cols)
    # 提取X和Y标签
    cols = [col for col in df.columns if col not in ['USER_ID', 'CFCS_Label', 'CFCS']]
    X = df[cols]
    y = df['CFCS_Label']
    print(len(X))
    return X, y


def train_by_xgb():
    print("training xgb model...")
    # 处理类别不均衡问题
    data = pd.read_csv('data/credit_pri_data.csv')
    X, y = process_for_classification(df=data)
    # smote = SMOTE(random_state=42)
    # x_res, y_res = smote.fit_sample(X, y)
    # 3:1 将数据换分出训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    print(len(x_train), len(x_test), len(x_train) + len(x_test))
    clf = XGBClassifier(max_depth=6)
    clf.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric='logloss',
            verbose=True)

    # 在测试集验证效果
    print("evaluation on test data:")
    y_pred = clf.predict(x_test)
    label = pd.DataFrame()
    label['y_pred'] = y_pred
    print(label['y_pred'].value_counts())
    y_prob = clf.predict_proba(x_test)[:, 1]
    print(accuracy_score(y_test, y_pred))
    print(roc_auc_score(y_test, y_prob))
    print(y_prob)

    # 根据分类概率得出用户评分
    print("forming the final result")
    all_data = pd.read_csv('data/credit_pri_data.csv')
    x_all, _ = process_for_classification(df=all_data)
    all_data['label'] = clf.predict_proba(x_all)[:, 0]
    all_data['label'] = all_data['label'].apply(lambda x: round((x * 100 - 90) * 9.7, 2))
    all_data[['USER_ID', 'label']].to_csv('xgb_submission.csv', index=None)

    # from xgboost import plot_importance
    # from matplotlib import pyplot
    #
    # plot_importance(clf, height=0.5, max_num_features=15)
    # pyplot.show()


train_by_xgb()


def train_by_kmeans():
    data = pd.read_csv('data/credit_pri_data.csv')
    cols = [col for col in data.columns if col not in ['USER_ID']]
    clf = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                 n_clusters=10, n_init=10, n_jobs=1, precompute_distances='auto',
                 random_state=42, tol=0.0001, verbose=0)
    clf.fit(data[cols])
    clf = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                 n_clusters=10, n_init=10, n_jobs=1, precompute_distances='auto',
                 random_state=42, tol=0.0001, verbose=0)
    clf.fit(data[cols])
    data['label'] = clf.labels_  # 对原数据表进行类别标记
    data['label'].value_counts().plot(kind='barh')
    plt.show()
    data['label'] = data['label'].apply(lambda x: (10 - x) * 9)
    data[['USER_ID', 'label']].to_csv('submission.csv', index=None)

train_by_kmeans()