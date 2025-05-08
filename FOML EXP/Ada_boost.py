from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost model with decision tree as base learner
base_model = DecisionTreeClassifier(max_depth=1)  # Stump as weak classifier
ada_boost = AdaBoostClassifier(base_model, n_estimators=50)
ada_boost.fit(X_train, y_train)

# Predict and evaluate
y_pred = ada_boost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix - AdaBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance visualization
feature_importances = ada_boost.feature_importances_
plt.barh(data.feature_names, feature_importances)
plt.title("Feature Importances - AdaBoost")
plt.show()
