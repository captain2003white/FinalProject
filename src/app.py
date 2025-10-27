# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 读取数据集
df = pd.read_csv('data/train_and_test2.csv', encoding='utf-8')

# 删除zero列
for i in range(0, 19):
    if i == 0:
        df.drop(['zero'], inplace=True, axis=1)
    else:
        df.drop([f'zero.{i}'], inplace=True, axis=1)

# 对缺失值进行向前填充
df['Embarked'].fillna(method='ffill', inplace=True)

# 建立标签
X = df.drop(['Passengerid', '2urvived'], axis=1)
y = df['2urvived']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

# 实例化SVM
svm = SVC(C=1, gamma=0.1, kernel='rbf')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

# 计算混淆矩阵
confusion = confusion_matrix(y_test, y_pred)
print("原始混淆矩阵：")
print(confusion)

# 归一化混淆矩阵
confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
print("\n归一化混淆矩阵：")
print(confusion_normalized)

# 计算模型评价指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'\naccuracy={accuracy:.4f}, recall={recall:.4f}, precision={precision:.4f}, f1={f1:.4f}')

# 打印详细的分类报告
from sklearn.metrics import classification_report
print("\n详细分类报告：")
print(classification_report(y_test, y_pred, target_names=['负类', '正类']))