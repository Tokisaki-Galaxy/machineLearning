import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('blood_pressure.csv')

# 选择特征和目标变量
X = data[['Age', 'BMI', 'Cholesterol']]
y = data['BP']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"最初模型评估:")
print(f"均方误差(MSE): {mse:.2f}")
print(f"R平方(R2): {r2:.2f}")

# 方法1：特征缩放(虽然线性回归通常不需要，但有时有帮助)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 重新训练模型
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)

# 评估缩放后的模型
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print(f"\n缩放后模型评估:")
print(f"均方误差(MSE): {mse_scaled:.2f}")
print(f"R平方(R2): {r2_scaled:.2f}")

# 方法2：尝试多项式特征(如果线性关系不明显)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"\n多项式模型评估:")
print(f"均方误差(MSE): {mse_poly:.2f}")
print(f"R平方(R2): {r2_poly:.2f}")