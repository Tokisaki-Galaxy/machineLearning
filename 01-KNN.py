import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

csvdata = pd.read_csv('medical_device_adverse_events.csv')
X=csvdata.drop(columns=['Adverse_Event'])
y=csvdata['Adverse_Event']

# 标准化 划分
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k=7
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 评估
print(f"当k={k}\n模型评估:")
print("准确率:", accuracy_score(y_test, y_pred))
print("召回率:", recall_score(y_test, y_pred, average='weighted'))
print("F1分数:", f1_score(y_test, y_pred, average='weighted'))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

print("\n不同K值下的F1:")
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    f1s = f1_score(y_test, y_pred, average='weighted')
    print(f"K={k}: F1={f1s:.4f}")