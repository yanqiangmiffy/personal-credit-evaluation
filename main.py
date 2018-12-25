# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: main.py 
@Time: 2018/12/5 11:43
@Software: PyCharm 
@Description:
"""
from __future__ import print_function
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def process_for_xgb(df):
    """
    分类模型 数据预处理
    :return:
    """
    print("proces raw data for xgb...")
    # 去除用户USER_ID 为空的数据
    df.dropna(subset=['USER_ID'], inplace=True, axis=0)
    # 使用众数填充AGE的缺失值
    df['AGE'] = df['AGE'].fillna(df['AGE'].median())
    # 使用0 填充Sale_houcs/JFCS和CFCF的缺失值
    df['Purchase_houcs'] = df['Purchase_houcs'].fillna(0).astype('int32')
    df['Sale_houcs'] = df['Sale_houcs'].fillna(0).astype('int32')
    df['JFCS'] = df['JFCS'].fillna(0).astype('int32')
    df['CFCS'] = df['CFCS'].fillna(0).astype('int32')  # 类别标签1
    df['CFCS_Label'] = df['CFCS'].apply(lambda x: 1 if x != 0 else 0)  # 类别标签1
    df['PROPERTY_DATE'] = df['PROPERTY_SIGN_DATE'].apply(lambda x: x.split('/')[-1]).astype('int32')
    # print(df['Mortgage_starttime'].count())
    # df['PROPERTY_DATE'].value_counts().plot(kind='barh')
    # plt.show()
    # 去除噪音的列或者无关紧要的列
    df.drop(
        columns=['SEX', 'HOU_ID', 'PROPERTY_SIGN_DATE', 'PROPERTY_RECORD_DATE', 'Mortgage_starttime',
                 'Mortgage_endtime'],
        inplace=True, axis=1)

    # 处理类别标签 one-hot
    cate_cols = ['TEL_ID', 'PROVINCE', 'NATIONALITY', 'PROPERTY_ID', 'PROPERTY_USAGE_TYPE', 'PROPERTY_LOAN_WAY',
                 'PROPERTY_PAYMENT', 'PROPERTY_DATE']
    df = pd.get_dummies(df, columns=cate_cols)
    # 提取X和Y标签
    cols = [col for col in df.columns if col not in ['USER_ID', 'CFCS_Label', 'CFCS', 'JFCS']]
    X = df[cols]
    y = df['CFCS_Label']
    print(X.shape)
    return X, y


def train_by_xgb():
    data = pd.read_csv('data/credit_final_data.csv')
    X, y = process_for_xgb(df=data)
    print("training xgb model...")
    # 处理类别不均衡问题
    smote = SMOTE(random_state=42)
    x_res, y_res = smote.fit_sample(X.values, y.values)
    # 3:1 将数据换分出训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_res, y_res)
    clf = XGBClassifier(learning_rate=0.1,
                        n_estimators=100, silent=True,
                        objective="binary:logistic",
                        booster='gbtree',
                        max_depth=6)
    clf.fit(x_train, y_train,
            eval_set=[(x_test, y_test)],
            eval_metric='logloss',
            verbose=True)

    # 在测试集验证效果
    print("evaluation on test data:")
    y_pred = clf.predict(x_test)
    label = pd.DataFrame()
    label['y_pred'] = y_pred
    y_prob = clf.predict_proba(x_test)[:, 1]
    print("Evaluation on test precision ：%s" % accuracy_score(y_test, y_pred))
    print("Evaluation on test auc score  ：%s" % roc_auc_score(y_test, y_prob))

    # 根据分类概率得出用户评分
    print("forming the xgb final result")
    all_data = pd.read_csv('data/credit_final_data.csv')
    x_all, _ = process_for_xgb(df=all_data)
    all_data['label'] = clf.predict_proba(x_all.values)[:, 0]
    all_data['label'] = all_data['label'].apply(lambda x: round(x * 95, 2))
    # all_data[['USER_ID', 'label']].to_csv('xgb_submission.csv', index=None)

    # from xgboost import plot_importance
    # from matplotlib import pyplot
    #
    # plot_importance(clf, height=0.5, max_num_features=15)
    # pyplot.show()

    result = all_data[['USER_ID', 'label']]
    return result


def process_for_kmeans(df):
    print("proces raw data for k-means...")
    # 去除用户USER_ID 为空的数据
    df.dropna(subset=['USER_ID'], inplace=True, axis=0)

    # 使用平均值填充AGE的缺失值
    df['AGE'] = df['AGE'].fillna(df['AGE'].mean())
    # 使用0 填充Sale_houcs/JFCS和CFCF的缺失值
    df['Purchase_houcs'] = df['Purchase_houcs'].fillna(0).astype('int32')
    df['Sale_houcs'] = df['Sale_houcs'].fillna(0).astype('int32')
    df['JFCS'] = df['JFCS'].fillna(0).astype('int32')
    df['CFCS'] = df['CFCS'].fillna(0).astype('int32')  # 类别标签1

    # 去除噪音的列或者无关紧要的列
    df.drop(
        columns=['SEX', 'HOU_ID', 'PROPERTY_SIGN_DATE', 'PROPERTY_RECORD_DATE', 'Mortgage_starttime',
                 'Mortgage_endtime'],
        inplace=True, axis=1)

    # 处理类别标签 one-hot
    cate_cols = ['TEL_ID', 'PROVINCE', 'NATIONALITY', 'PROPERTY_ID', 'PROPERTY_USAGE_TYPE', 'PROPERTY_LOAN_WAY',
                 'PROPERTY_PAYMENT']
    df = pd.get_dummies(df, columns=cate_cols)
    return df


def train_by_kmeans():
    data = pd.read_csv('data/credit_final_data.csv')
    data = process_for_kmeans(df=data)
    print("training k-means")
    clf = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                 n_clusters=10, n_init=10, n_jobs=1, precompute_distances='auto',
                 random_state=42, tol=0.0001, verbose=0)
    # 去除USER_ID一列
    cols = [col for col in data.columns if col not in ['USER_ID']]
    clf.fit(data[cols])
    clf = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                 n_clusters=10, n_init=10, n_jobs=1, precompute_distances='auto',
                 random_state=42, tol=0.0001, verbose=0)
    clf.fit(data)
    data['label'] = clf.labels_  # 对原数据表进行类别标记
    # 按照人数从低到高排序
    label_score = dict()
    as_labels = data['label'].value_counts(ascending=True).index.tolist()
    print(as_labels)
    for index, label in enumerate(as_labels):
        label_score[label] = (index + 1) * 9

    data['label'].value_counts().plot(kind='barh')
    plt.show()
    data['label'] = data['label'].apply(lambda x: label_score[x])
    # data[['USER_ID', 'label']].to_csv('submission.csv', index=None)

    result = data[['USER_ID', 'label']]
    return result


def main():
    """
    根据xgb分类模型以及kmeans聚类结果，降权求和
    :return:
    """
    result_xgb = train_by_xgb()
    result_kmeans = train_by_kmeans()
    result_xgb.drop_duplicates(subset=['USER_ID'], keep='first', inplace=True)
    result_kmeans.drop_duplicates(subset=['USER_ID'], keep='first', inplace=True)

    print(len(result_xgb))
    print(len(result_kmeans))

    result = pd.merge(result_xgb, result_kmeans, how='inner', on='USER_ID')
    result['label'] = result['label_x'] * 0.7 + result['label_y'] * 0.3
    # result.to_csv('result.csv')

    result['USER_ID'] = result['USER_ID'].astype('int32')
    result['label'] = result['label'].apply(lambda x: round(x, 2))
    result['label'] = result['label'].apply(lambda x: round(x, 2))
    result.rename(columns={'USER_ID': '用户ID', 'label': '房产信用评分'}, inplace=True)
    result[['用户ID', '房产信用评分']].to_csv('User_Credit_Predict.csv', index=False)


if __name__ == '__main__':
    main()
