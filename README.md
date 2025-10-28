# 🎙️ Voice Gender Classification

This project classifies **human voices as Male or Female** using extracted audio features. It involves end-to-end preprocessing, outlier handling, feature selection, clustering with KMeans, model training with multiple classifiers, performance evaluation, and experiment tracking using **MLflow**.

---

## 🧠 Project Overview

The aim of this project is to build a **robust machine learning classification model** that predicts the **gender of a speaker’s voice** based on acoustic features. This helps in applications like voice recognition, biometric authentication, and speech analysis.

---

## 🧩 Problem Statement

Given numerical acoustic features extracted from human voice recordings, predict whether the voice belongs to a **male (0)** or **female (1)** speaker.

---

## 🗂️ Dataset Details

* **File:** `vocal_gender_features_new.csv`
* **Samples:** 16,148
* **Features:** 44 (including the label)

| Category | Example Features                                         |
| -------- | -------------------------------------------------------- |
| Spectral | mean_spectral_centroid, spectral_skew, spectral_kurtosis |
| Energy   | rms_energy, log_energy, energy_entropy                   |
| MFCCs    | mfcc_1_mean … mfcc_13_std                                |
| Pitch    | mean_pitch, min_pitch, max_pitch, std_pitch              |
| Target   | label (0 = Male, 1 = Female)                             |

---

## 🧹 Data Preprocessing

1. **Duplicate Removal:** 1,078 duplicate rows removed
2. **Missing Values:** None found
3. **Outlier Treatment:** Capped using the IQR method
4. **Feature Selection:** Removed low-variance features (`VarianceThreshold=0.01`)
5. **Feature Scaling:** Standardized using `StandardScaler`
6. **KMeans Clustering:** Added cluster labels (k=2) as an additional feature

---

## ⚙️ Feature Engineering

Outliers were handled using the Interquartile Range (IQR) method:

```python
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)
```

---

## 🤖 Model Building

Several machine learning algorithms were trained and compared:

| Model                        | Accuracy   |
| ---------------------------- | ---------- |
| Logistic Regression          | 99.17%     |
| K-Nearest Neighbors          | 99.83%     |
| Support Vector Machine (RBF) | **99.86%** |
| Decision Tree                | 95.45%     |
| Random Forest                | 99.40%     |
| Gradient Boosting            | 99.17%     |
| Naive Bayes                  | 90.80%     |

✅ **Best Model:** **SVM (RBF Kernel)** with **99.86% accuracy**

---

## 📊 Model Evaluation

Example confusion matrix and classification report:

```
Confusion Matrix:
[[1033   13]
 [  12 1956]]

Precision, Recall, F1-score:
 0 -> Precision: 0.99, Recall: 0.99, F1: 0.99  
 1 -> Precision: 0.99, Recall: 0.99, F1: 0.99
```

---

## 🔍 MLflow Experiment Tracking

All models were tracked using **MLflow**, including their:

* Parameters
* Accuracy, precision, recall, F1-score
* Logged models for reproducibility

Example:

```python
mlflow.set_experiment("Voice Gender Classification")

with mlflow.start_run(run_name="SVM"):
    mlflow.sklearn.log_model(model, "SVM")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_params(model.get_params())
```

---

## 💾 Model Saving

The **best model (Random Forest)** and **scaler** were saved for deployment:

```python
import joblib
joblib.dump(best_model, "voice_gender_model.pkl")
joblib.dump(scaler, "voice_prediction.pkl")
```

---

## 📈 Insights

* Data slightly imbalanced (65% female, 35% male)
* MFCC features play a major role in classification
* SVM and KNN handle feature scaling effectively

---

## 🧰 Technologies Used

* **Python 3.10+**
* **Libraries:**
  `pandas`, `numpy`, `seaborn`, `matplotlib`,
  `scikit-learn`, `mlflow`, `joblib`

---

## 🚀 Future Work

* Build a **Streamlit Web App** for interactive gender prediction
* Apply **Deep Learning (CNN / LSTM)** for improved accuracy
* Integrate **real-time audio feature extraction**

---

## 👨‍💻 Author

**Sachin Hembram**
Data Science Enthusiast | Machine Learning Developer

---
