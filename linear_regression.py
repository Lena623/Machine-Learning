import numpy as np
import pandas as pd

#
data=pd.read_csv(r'D:\TemporaryFiles\Machine-learning\LinearRegression\student-mat.csv',sep=';')

"""
# 看看数据长什么样
print("数据前5行：")
print(data.head())

print("\n数据有多少行多少列：")
print(data.shape)

#看看有哪些特征)
print("\n所有列名：")
print(data.columns.tolist())
"""

# Extract features X and target y
# Use studytime, failures, and absences to predict (G3
X = data[['studytime', 'failures', 'absences']].values
y = data['G3'].values

# normalization
x = (X - X.mean(axis = 0)) / X.std(axis = 0)

# initialization
m,n = X.shape
w = np.zeros(n)
b = 0.0
learning_rate = 0.01
epochs = 1000

# gradient descent iteration
for epoch in range(epochs):
    y_pred = X @ w + b
    error = y_pred - y

    dw = (1 / m) * (X.T @ error)
    db = (1 / m) * np.sum(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    # 每100轮打印一次当前的损失值 (MSE)
    if (epoch + 1) % 100 == 0:
        loss = (1 / (2 * m)) * np.sum(error ** 2)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

print("\n训练完成！")
print("最终权重 w:", w)
print("最终偏置 b:", b)

# predict
new_student = np.array([[0.5,-0.5,0.2]])
prediction = new_student @ w + b
print(f"Predict the final grade:{prediction.item():.2f}")