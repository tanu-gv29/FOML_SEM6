from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# Generate dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, n_informative=2, random_state=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Perceptron(max_iter=1000, eta0=0.01)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Greens', fmt='d')
plt.title("Confusion Matrix - Single Layer Perceptron")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualization
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='o')
plt.title("Predicted Classes - Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
