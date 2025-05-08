import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generating synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Flattening arrays
X = X.flatten()
y = y.flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Least Squares Method
mean_x, mean_y = np.mean(X_train), np.mean(y_train)
m = sum((X_train - mean_x) * (y_train - mean_y)) / sum((X_train - mean_x)**2)
c = mean_y - m * mean_x

# Predict
y_pred = m * X_test + c

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Slope (m):", round(m, 2))
print("Intercept (c):", round(c, 2))
print("MSE:", round(mse, 2))
print("R² Score:", round(r2, 2))

# Plot
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.title("Simple Linear Regression (Least Squares)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
