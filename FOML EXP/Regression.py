import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from mpl_toolkits.mplot3d import Axes3D

# Load California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target  # Use median house value as target

# Function to train and evaluate regression
def run_regression(features, label='PRICE', title=''):
    X = df[features].values
    y = df[label].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {title} ---")
    print(f"Features used: {features}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Visualization
    if X_test.shape[1] == 1:
        # --- Univariate: 2D Plot ---
        plt.figure(figsize=(8, 5))
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
        plt.title(f"{title} Regression")
        plt.xlabel(features[0])
        plt.ylabel("PRICE")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif X_test.shape[1] == 2:
        # --- Bivariate: 3D Plot ---
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', label='Actual')
        ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, c='red', label='Predicted', alpha=0.6)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel('PRICE')
        plt.title(f"{title} Regression (3D View)")
        ax.legend()
        plt.show()

    else:
        # --- Multivariate: Actual vs Predicted Plot ---
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        plt.title(f"{title} Regression - Actual vs Predicted")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.grid(True)
        plt.show()

# Run All Types of Regression with Visualization

run_regression(['MedInc'], title='Univariate')

run_regression(['MedInc', 'AveRooms'], title='Bivariate')

run_regression(['MedInc', 'AveRooms', 'AveOccup', 'HouseAge', 'Latitude', 'Longitude'], title='Multivariate')
