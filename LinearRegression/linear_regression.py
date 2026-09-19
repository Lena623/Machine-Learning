import numpy as np
import pandas as pd

data=pd.read_csv('LinearRegression/student-mat.csv',sep=';')

# Extract features X and target y
# Use studytime, failures, and absences to predict (G3
X = data[['studytime', 'failures', 'absences']].values
y = data['G3'].values

# normalization
X = (X - X.mean(axis = 0)) / X.std(axis = 0)

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

    # Calculate gradient
    dw = (1 / m) * (X.T @ error)
    db = (1 / m) * np.sum(error)

    # update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Print the current loss value (MSE) every 100 epochs.
    if (epoch + 1) % 100 == 0:
        loss = (1 / (2 * m)) * np.sum(error ** 2)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

print("\nTraining Finished")
print("Optimal weight w:", w)
print("Optimal bias b:", b)

# predict
new_student = np.array([[0.5,-0.5,0.2]])
prediction = new_student @ w + b
print(f"Predict the final grade:{prediction.item():.2f}")