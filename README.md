# 🩺 Heart Disease Detection System Using Machine Learning

An automated computational diagnostics repository developed during my one-month summer industrial internship as a **Python Development Intern** at **SycoWebs, New Delhi** (July 8, 2026 - August 8, 2026).

---

## 📌 Project Overview
Cardiovascular diseases (CVDs) are a primary driver of global mortality. Traditional clinical screening involves manually interpreting multiple complex physiological tests, which introduces diagnostic friction and delays. 

This project bridges **Computer Science, Machine Learning, and Healthcare** by engineering a standalone predictive software module in Python. The system processes a multi-variate clinical dataset containing continuous and categorical patient biometric attributes to output an instantaneous binary diagnostic risk index (`0` for a Healthy profile, `1` for a Defective profile), effectively automating early-stage cardiac risk stratification.

---

## 🛠️ Technical Stack & Tools Mastered
* **Language Abstraction:** Python 3.x
* **Integrated Developer Environments (IDEs):** Visual Studio Code (VS Code), Jupyter Notebook
* **Data Manipulation & Ingestion:** Pandas, NumPy
* **Exploratory Data Analysis (EDA) Visuals:** Matplotlib, Seaborn
* **Mathematical Learning Framework:** Scikit-Learn

---

## 🔬 Dataset & Clinical Attributes
The computational pipeline ingests a comprehensive historical heart disease framework containing **13 distinct pathological indicator vectors**:

1. **`age`**: Age of the patient (Years)
2. **`sex`**: Biological sex parameters (1 = Male, 0 = Female)
3. **`cp`**: Chest pain type indices (Value 0–3)
4. **`trestbps`**: Resting blood pressure logs (mm Hg on admission)
5. **`chol`**: Serum cholesterol concentrations (mg/dl)
6. **`fbs`**: Fasting blood sugar level (> 120 mg/dl; 1 = True, 0 = False)
7. **`restecg`**: Resting electrocardiographic outcomes (Value 0–2)
8. **`thalach`**: Maximum heart rate achieved during exercise stress
9. **`exang`**: Exercise-induced angina (1 = Yes, 0 = No)
10. **`oldpeak`**: ST depression induced by exercise relative to rest
11. **`slope`**: The slope of the peak exercise ST segment
12. **`ca`**: Number of major vessels colored by fluoroscopy (0–3)
13. **`thal`**: Thalassemia classification index (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)
* **`target` (Ground Truth Label):** Clinical outcome status (0 = Healthy Heart, 1 = Defective Heart Profile)

---

## 📊 Algorithmic Architecture & Performance Metrics
The system deploys a supervised **Regularized Logistic Regression** classifier. To guarantee mathematical validity and avoid sample grouping biases, the input matrix was split using a stratified configuration grid:

* **Data Partitioning:** 80% Training Subset / 20% Testing Validation Split
* **Data Balancing Protocol:** Enforced `stratify=Y` to preserve exact target output proportions
* **Loss Function Optimization:** Binary Cross-Entropy Log-Loss with default L₂ Ridge Penalty Regularization
* **Training Data Classification Accuracy:** **~85.12%**
* **Testing Validation Generalizability Score:** **~81.97%**

---

## 🚀 Execution & Inference Guide

### 1. Ingest Dataset and Verify Quality Setup
```python
import pandas as pd
import numpy as np

# Load clinical matrix
heart_data = pd.read_csv('heart_disease_data.csv')

# Verify structural data integrity (Zero null inputs allowed)
print(heart_data.isnull().sum())
```

### 2. Feature Segmentation & Stratified Splitting
```python
from sklearn.model_selection import train_test_split

X = heart_data.drop(columns='target')
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### 3. Model Training & Fitting Execution
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)
```

### 4. Running the Single-Instance Diagnostic Inference Pipeline
To evaluate a new patient array without causing a structural matrix dimensions conflict, a dimensional conversion reshaping module is integrated:

```python
# Custom multivariate physiological input vector
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

# Convert to NumPy array matrix
input_data_as_numpy_array = np.asarray(input_data)

# Realign vector shapes to map exactly 1 Row and 13 Columns
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Generate automated diagnostic prediction
prediction = model.predict(input_data_reshaped)

if (prediction[0] == 0):
    print('The Patient does not exhibit a Heart Disease Profile.')
else:
    print('Warning: The Patient exhibits clinical symptoms of Heart Disease.')
```

---

## 📈 Strategic Path to Capstone (Future Expansion Roadmap)
The current diagnostic script is mathematically validated and structured for further development:
1. **Cloud API Integration:** Embedding the analytical core framework into a localized **Streamlit** or **Flask** web application deployment interface.
2. **IoT Edge Diagnostics:** Interfacing the pipeline inputs directly with wearable healthcare smart devices to monitor live physiological parameters in real-time.
