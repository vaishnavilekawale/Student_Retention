# 🎓 Student Retention Analysis Using Machine Learning

## 📌 Project Overview

This project predicts whether a student is likely to **drop out or continue** based on academic and personal factors such as:

* Attendance
* CGPA
* Backlogs
* Participation
* Family Income

The system uses multiple Machine Learning models and selects the best performing model based on accuracy.

---

## 🚀 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📊 Dataset Information

The dataset is synthetically generated using NumPy with 500 student records.

### Features:

| Feature       | Description                                 |
| ------------- | ------------------------------------------- |
| Attendance    | Student attendance percentage               |
| CGPA          | Cumulative Grade Point Average              |
| Backlogs      | Number of failed subjects                   |
| Participation | Extra-curricular participation score        |
| Family_Income | Family income level                         |
| Dropout       | Target variable (1 = Dropout, 0 = Continue) |

Dropout condition is artificially defined as:

* Attendance < 60
* OR CGPA < 5.0
* OR Backlogs > 3

---

## 🧠 Machine Learning Models Used

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)

The model with the highest accuracy is automatically selected as the **Best Model**.

---

## 📈 Project Workflow

1. Dataset Generation
2. Data Visualization
3. Data Preprocessing
4. Model Training
5. Model Evaluation
6. Best Model Selection
7. Confusion Matrix & Classification Report
8. Manual Prediction System

---

## 📊 Visualizations Included

* Dropout Distribution Count Plot
* Attendance vs CGPA Scatter Plot
* Confusion Matrix Heatmap

---

## 🎯 Results

* Multiple models were compared based on accuracy.
* The best performing model was selected automatically.
* Classification report provides precision, recall, and F1-score.

---

## 💻 How to Run This Project

### Step 1: Clone the Repository

```
https://github.com/vaishnavilekawale/Student_Retention
```

### Step 2: Install Required Libraries

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Run the Python File

```
python student_retention.py
```

---

## 🔮 Manual Prediction System

After training, the system allows user input:

* Attendance
* CGPA
* Backlogs
* Participation
* Family Income

It predicts whether the student is likely to:

✅ Continue
⚠ Drop Out

---

## 👩‍💻 Author

Vaishnavi Lekawale

---

## ⭐ If you like this project

Give it a star on GitHub!
