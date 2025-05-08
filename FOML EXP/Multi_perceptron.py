import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset from file
file_path = "banknote-authentication.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Step 2: Preprocess the dataset (Check for missing values)
print(data.info())
print(data.describe())

# Step 3: Prepare the data (Assuming last column is 'Class' and rest are features)
X = data.iloc[:, :-1].values  # Features (all columns except last)
y = data.iloc[:, -1].values   # Target (last column)

# Step 4: Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Define the MLP model (1 hidden layer with 10 neurons)
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Step 7: Train the model
mlp.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = mlp.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2%}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Step 10: Test the model with a new sample
new_sample = [[2.5, -1.2, 3.1, -0.8]]  # Replace with actual feature values
new_sample_scaled = scaler.transform(new_sample)
prediction = mlp.predict(new_sample_scaled)
print(f"Predicted Class: {'Forged' if prediction[0] == 1 else 'Genuine'}")