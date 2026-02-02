# Phishing URL Detection System 

###  Project Overview
Phishing attacks are one of the most common security threats today. This project focuses on building a machine learning system to automatically classify URLs as **Phishing (Malicious)** or **Legitimate (Safe)**.

By analyzing specific features of URLs (such as domain characteristics and structural patterns), this project compares eight different classification algorithms to determine the most effective model for real-time threat detection.

### Objective
*   To analyze the **PhiUSIIL Phishing URL Dataset**.
*   To perform data preprocessing, including encoding and scaling.
*   To compare the performance of various machine learning models (Linear, Tree-based, and Ensemble methods).
*   To identify the best-performing model based on **F1-Score** and **Recall** (minimizing false negatives is crucial in security).

---

###  Technologies Used
*   **Language:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn, XGBoost
*   **Environment:** Jupyter Notebook / Python Script

---

###  Dataset
The project uses the **PhiUSIIL Phishing URL Dataset**.
*   **Target Variable:** `label` (0 = Legitimate, 1 = Phishing)
*   **Features:** Includes domain information, TLDs, and various extracted URL characteristics.
*   *Note: Unnecessary identifiers like Filename, URL string, and Title were removed during the cleaning process.*

---

###  Methodology

#### 1. Exploratory Data Analysis (EDA)
*   Checked for class imbalance to ensure the dataset represents both phishing and safe URLs adequately.
*   Analyzed data shape and missing values.

#### 2. Feature Engineering & Preprocessing
Data preparation was tailored to the specific requirements of different algorithms:
*   **Categorical Encoding:** `Domain` and `TLD` columns were Label Encoded.
*   **Numerical Scaling:** Applied `StandardScaler` to numerical features to normalize the data distribution.
*   **Model-Specific Processing:**
    *   *Tree-based models (Random Forest, XGBoost, Decision Tree):* Used encoded data *without* scaling (to preserve interpretability).
    *   *Distance/Gradient-based models (SVM, KNN, Logistic Regression, MLP):* Used fully *scaled* data to ensure convergence and accuracy.

#### 3. Model Training
The following algorithms were trained and evaluated:
1.  Logistic Regression
2.  Decision Tree
3.  Random Forest
4.  Support Vector Machine (SVM - RBF Kernel)
5.  k-Nearest Neighbors (KNN)
6.  Gaussian Naive Bayes
7.  XGBoost Classifier
8.  MLP Classifier (Neural Network)

---

### Results & Performance
The models were evaluated using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | *0.9* | *0.9* | *0.9* | *0.907158* | *0.971249* |
| **Random Forest** | *0.97* | *0.97* | *0.97* | *0.965282* | *0.992559* |


#### Key Findings:
*   **Random Forest** achieved the highest overall performance.
*   **Confusion Matrices** were generated to analyze where models struggled (e.g., False Positives vs. False Negatives).
*   Tree-based models generally performed better on this dataset compared to linear models.

---

###  How to Run
1.  Clone the repository:
    ```bash
    git clone https://github.com/RLUDHIYA/PhishingDetectionModel.git
    ```
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
3.  Ensure the dataset `PhiUSIIL_Phishing_URL_Dataset.csv` is in the root directory.
4.  Run the script:
    ```bash
    python phishing_detection.py
    ```

---

### Future Improvements
*   **Hyperparameter Tuning:** Implementing `GridSearchCV` to optimize model parameters.
*   **Feature Selection:** Using Recursive Feature Elimination (RFE) to reduce dimensionality.

---

###  Author
**Ludhiya Rose Giji**
*   *Aspiring Data Analyst*
*   [(Ludhiya LinkedIn)](https://www.linkedin.com/in/ludhiya-rose-giji-0a521b22a/)
  

---
