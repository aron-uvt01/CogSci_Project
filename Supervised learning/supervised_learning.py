
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Load and merge data 
eeg_data = pd.read_csv("/Users/butararon/Desktop/Uvt-Cognitive Science/Year 1 /Semester 2 /Computer science and cognition /Projects/Dots_30_006_data.csv")
behavioral_data = pd.read_csv("/Users/butararon/Desktop/Uvt-Cognitive Science/Year 1 /Semester 2 /Computer science and cognition /Projects/Dots_30_006_trial_info.csv")

data = eeg_data.copy()
data['GForce'] = behavioral_data['GForce']
data['ResponseID'] = behavioral_data['ResponseID']


X = data.drop(columns=['Trial', 'ResponseID'])
y = data['ResponseID']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

#Random forest
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X_train, y_train)
base_preds = base_model.predict(X_test)
base_acc = accuracy_score(y_test, base_preds)

#Confusion matrix pentru random forest 
ConfusionMatrixDisplay.from_estimator(base_model, X_test, y_test)
plt.title(f"Base RF Confusion Matrix (Acc = {base_acc:.2f})")
plt.tight_layout()
plt.savefig("base_rf_confusion_matrix.png")
plt.show()

#Aici am bagat hill climbing pentru random forest
best_model = base_model
best_score = base_acc
best_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2
}

random.seed(42)
for step in range(20):
    candidate = best_params.copy()
    candidate["n_estimators"] = max(10, candidate["n_estimators"] + random.choice([-50, 0, 50]))
    candidate["max_depth"] = random.choice([None, 5, 10, 20, 50])
    candidate["min_samples_split"] = random.choice([2, 5, 10])

    model = RandomForestClassifier(
        n_estimators=candidate["n_estimators"],
        max_depth=candidate["max_depth"],
        min_samples_split=candidate["min_samples_split"],
        random_state=42
    )
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))

    if score > best_score:
        best_score = score
        best_model = model
        best_params = candidate
        print(f"[Step {step}] New best score: {best_score:.3f} with {best_params}")

#Evaluare finala 
print("Final Accuracy:", best_score)
print("Best Parameters:", best_params)
print("Classification Report:")
print(classification_report(y_test, best_model.predict(X_test)))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"Hill Climbing RF Confusion Matrix (Acc = {best_score:.2f})")
plt.tight_layout()
plt.savefig("hillclimb_rf_confusion_matrix.png")
plt.show()

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

#Evaluarea
nb_accuracy = accuracy_score(y_test, nb_preds)
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")
print("Classification Report (Naive Bayes):")
print(classification_report(y_test, nb_preds))

#Yeah man confusion matrix
ConfusionMatrixDisplay.from_estimator(nb_model, X_test, y_test)
plt.title(f"Naive Bayes Confusion Matrix (Acc = {nb_accuracy:.2f})")
plt.tight_layout()
plt.savefig("naive_bayes_confusion_matrix.png")
plt.show()

#K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_preds)
print(f"KNN Accuracy: {knn_accuracy:.2f}")
print("Classification Report (KNN):")
print(classification_report(y_test, knn_preds))

ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test)
plt.title(f"KNN Confusion Matrix (Acc = {knn_accuracy:.2f})")
plt.tight_layout()
plt.savefig("knn_confusion_matrix.png")
plt.show()

#Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print("Classification Report (SVM):")
print(classification_report(y_test, svm_preds))

ConfusionMatrixDisplay.from_estimator(svm_model, X_test, y_test)
plt.title(f"SVM Confusion Matrix (Acc = {svm_accuracy:.2f})")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png")
plt.show()

model_scores = {
    "Random Forest": base_acc,
    "HillClimb RF": best_score,
    "Naive Bayes": nb_accuracy,
    "KNN": knn_accuracy,
    "SVM": svm_accuracy
}
plt.figure(figsize=(8, 5))
plt.bar(model_scores.keys(), model_scores.values(), color='skyblue')
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

#Aici vedem care feature e mai relevant
feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
feature_importance.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("top_features_rf.png")
plt.show()
