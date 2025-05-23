# Diabetes Classification using Machine Learning

## Project Overview

This project focuses on **diabetes classification** using **machine learning techniques** to analyze patient data and predict diabetes risk. The goal is to develop an **efficient, scalable model** to assist healthcare providers in early diabetes detection and risk assessment.

## Problem Statement

**Diabetes** is a chronic condition that, if not diagnosed and managed early, can lead to severe complications such as **kidney failure**, **blindness**, and **cardiovascular diseases**. Traditional diagnostic methods rely on manual interpretation of medical data, which can be **time-consuming** and **prone to errors**. Additionally, identifying at-risk individuals at an early stage remains challenging due to the complexity of the underlying factors.

### Solution

* Develop a **machine learning-based system** to classify diabetes risk using patient data such as **glucose levels**, **BMI**, **age**, and other relevant features.
* **Improve the accuracy** of diabetes classification through advanced ML models.
* Facilitate **early detection** for timely interventions.
* Build an **efficient and scalable model** for healthcare providers to use in diabetes risk assessment.

## Metadata

* **Each record** represents a unique patient.
* **Columns in dataset:**

  * `PhysActivity`: Physical activity level.
  * `Education`: Level of education attained.
  * `Income`: Income level of the patient.
  * `Veggies`: Frequency of vegetable consumption.
  * `Sex`: Gender of the patient.
  * `Fruits`: Frequency of fruit consumption.
  * `Smoker`: Smoking status.
  * `MentHlth`: Mental health status.
  * `HighChol`: Presence of high cholesterol.
  * `PhysHlth`: Physical health status.
  * `Age`: Age of the patient.
  * `BMI`: Body Mass Index.
  * `HighBP`: Presence of high blood pressure.
  * `GenHlth`: General health status.
  * `Diabetes_012`: Target variable (0: No diabetes, 1: Pre-diabetes, 2: Diabetes).
* **Dataset Size:** 100,000 records, 21 features.

## Implementation Steps

### 1. Data Preprocessing

* **Encoding:** Label encoding for categorical columns.
* **Duplicate Handling:** Removing duplicate records.
* **Missing Values:** Imputing missing data.
* **Class Imbalance:** Applying **SMOTE**.
* **Feature Extraction & Normalization:** Selecting important features and scaling numerical values.
* **Outlier Handling:** Identifying and managing outliers.

### 2. Machine Learning Models Used

* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **K-Means Clustering**
* **Decision Trees (DT)**
* **Random Forest (RF)**
* **Support Vector Machines (SVM)**
* **Naive Bayes**
* **Neural Networks (NN)**

## Expected Outcomes

* **High-accuracy ML models** for diabetes classification.
* **Efficient classification** using hyperparameter tuning and feature selection techniques.
* **Continuous model evaluation** with accuracy, precision, recall, and F1-score.
* **Feature importance analysis** to identify key risk factors.
* **Well-documented results** for academic research and healthcare innovation.

## Tools & Technologies

* **Python**
* **Pandas, NumPy** (Data manipulation & preprocessing)
* **Scikit-Learn** (ML models & evaluation)
* **Matplotlib, Seaborn** (Data visualization)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/lithigesh15/diabetes-classification.git
   ```
2. Navigate to the project directory:

   ```bash
   cd diabetes-classification
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the project:

   ```bash
   python main.py
   ```

## Future Enhancements

* Implementing **Deep Learning** models (e.g., Neural Networks).
* **Feature selection improvements** using advanced techniques.
* **Optimized hyperparameter tuning** for better accuracy.

---

### Thank you for exploring our project!

