# ============================================
# Student Retention Analysis Using ML
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------
# 1️⃣ Generate Sample Dataset
# --------------------------------------------

np.random.seed(42)

data_size = 500

data = pd.DataFrame({
    "Attendance": np.random.randint(40, 100, data_size),
    "CGPA": np.round(np.random.uniform(4.0, 10.0, data_size), 2),
    "Backlogs": np.random.randint(0, 6, data_size),
    "Participation": np.random.randint(0, 10, data_size),
    "Family_Income": np.random.randint(10000, 100000, data_size)
})

# Dropout Logic (Artificial Condition)
data["Dropout"] = np.where(
    (data["Attendance"] < 60) | 
    (data["CGPA"] < 5.0) | 
    (data["Backlogs"] > 3), 1, 0
)

print("Dataset Preview:\n")
print(data.head())

# --------------------------------------------
# 2️⃣ Data Visualization
# --------------------------------------------

plt.figure()
sns.countplot(x='Dropout', data=data)
plt.title("Dropout Distribution")
plt.show()

plt.figure()
sns.scatterplot(x='Attendance', y='CGPA', hue='Dropout', data=data)
plt.title("Attendance vs CGPA")
plt.show()

# --------------------------------------------
# 3️⃣ Data Preprocessing
# --------------------------------------------

X = data.drop("Dropout", axis=1)
y = data["Dropout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# 4️⃣ Model Training
# --------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

accuracy_results = {}

print("\nModel Results:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    
    print(f"{name} Accuracy: {acc:.4f}")

# --------------------------------------------
# 5️⃣ Best Model Selection
# --------------------------------------------

best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# --------------------------------------------
# 6️⃣ Confusion Matrix
# --------------------------------------------

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_best))

# --------------------------------------------
# 7️⃣ Manual Prediction System
# --------------------------------------------

print("\n=== Student Retention Prediction System ===")

attendance = float(input("Enter Attendance (40-100): "))
cgpa = float(input("Enter CGPA (0-10): "))
backlogs = int(input("Enter Number of Backlogs: "))
participation = int(input("Enter Participation Score (0-10): "))
income = int(input("Enter Family Income: "))

input_data = [[attendance, cgpa, backlogs, participation, income]]

prediction = best_model.predict(input_data)

if prediction[0] == 1:
    print("⚠ Student is likely to DROP OUT")
else:
    print("✅ Student is likely to CONTINUE")
