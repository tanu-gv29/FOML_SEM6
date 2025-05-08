import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Generate training data
X = []
y = []

for _ in range(1000):
    number = random.randint(1, 100)
    one_hot_input = [0] * 100
    one_hot_input[number - 1] = 1
    X.append(one_hot_input)
    y.append([number / 100])  # Normalize

X = np.array(X)
y = np.array(y)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

# Compile
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=30, batch_size=32, verbose=1)

# Function to get prediction
def guess_number_from_input():
    try:
        num = int(input("Enter a number between 1 and 100: "))
        if not 1 <= num <= 100:
            print("Please enter a valid number between 1 and 100.")
            return
        input_data = np.zeros((1, 100))
        input_data[0][num - 1] = 1
        prediction = model.predict(input_data)[0][0] * 100
        print(f"Actual Number: {num}")
        print(f"Model's Guess: {round(prediction, 2)}")
    except ValueError:
        print("Please enter a valid integer.")

# Run the function
guess_number_from_input()
