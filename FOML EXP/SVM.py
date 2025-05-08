from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset (LFW dataset for face recognition)
lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Preprocess data
X = lfw_people.data
y = lfw_people.target
X = StandardScaler().fit_transform(X)  # Feature scaling
pca = PCA(n_components=150)  # Reduce dimensionality for better performance
X_pca = pca.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=42)

# Train SVM model
svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Visualization
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix - SVM for Face Recognition")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
