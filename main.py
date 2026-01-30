import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report



pd.set_option('display.max_columns', None)

RANDOM_STATE = 42

# --- 1. Load the Dataset ---
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
print("Original DataFrame head:\n", df.head())

# --- 2. Initial EDA---
print("\nShape of the dataset:", df.shape)
# df.info() 

# Check for missing values
# print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False))

# Check class distribution
# print("\nClass distribution:\n", df['label'].value_counts())
# sns.countplot(data=df, x='label')
# plt.title("Phishing (1) vs Legitimate (0)")
# plt.show()

# --- 3. Feature Engineering/Selection ---
df.drop(columns=['FILENAME', 'URL', 'Title'], inplace=True, errors='ignore') 

# --- 4. Separate Features (X) and Target (y) ---
X = df.drop("label", axis=1)
y = df["label"]
print("\nFeatures (X) before preprocessing head:\n", X.head())
print("Target (y) head:\n", y.head())

# --- 5. Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_STATE,
                                                    stratify=y)

print("\nShapes after split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# --- 6. Preprocessing


X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# (a) Categorical Feature Encoding ('Domain', 'TLD')
categorical_cols = ['Domain', 'TLD']
label_encoders = {}
print("\n--- Encoding Categorical Features ---")
for col in categorical_cols:
    if col in X_train_processed.columns:
        print(f"Encoding column: {col}")
        le = LabelEncoder()
      
        X_train_processed[col] = le.fit_transform(X_train_processed[col])
      
        X_test_processed[col] = X_test_processed[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        label_encoders[col] = le 
    else:
        print(f"Warning: Categorical column '{col}' not found in X_train.")


X_train_encoded_unscaled = X_train_processed.copy()
X_test_encoded_unscaled = X_test_processed.copy()


# (b) Numerical Feature Scaling
numerical_cols_to_scale = [col for col in X_train_processed.columns if col not in categorical_cols]

print("\n--- Scaling Numerical Features ---")
if numerical_cols_to_scale:
    print("Numerical columns to be scaled:", numerical_cols_to_scale)
    scaler = StandardScaler()
    
    X_train_processed[numerical_cols_to_scale] = scaler.fit_transform(X_train_processed[numerical_cols_to_scale])
    
    X_test_processed[numerical_cols_to_scale] = scaler.transform(X_test_processed[numerical_cols_to_scale])
    print("Scaling complete.")
else:
    print("No numerical columns (other than encoded categoricals) identified for scaling.")


print("\nHead of X_train_processed (scaled & encoded):\n", X_train_processed.head())
print("Head of X_train_encoded_unscaled (encoded, unscaled numericals):\n", X_train_encoded_unscaled.head())


# --- 7. Model Definitions ---
models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes": GaussianNB(),
    "XGBoost": xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
    "MLP Classifier": MLPClassifier(random_state=RANDOM_STATE, max_iter=300, early_stopping=True, hidden_layer_sizes=(64,32), learning_rate_init=0.001)
}

results = {} 

# --- 8. Model Training & Evaluation Loop ---
print("\n--- Training and Evaluating Models ---")
for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")

  
    if model_name in ["Logistic Regression", "SVM (RBF Kernel)", "k-Nearest Neighbors", "MLP Classifier", "Gaussian Naive Bayes"]:
        current_X_train = X_train_processed
        current_X_test = X_test_processed
        print("Using SCALED & ENCODED data.")
    elif model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
        current_X_train = X_train_encoded_unscaled
        current_X_test = X_test_encoded_unscaled
        print("Using ENCODED (but unscaled numericals) data.")
    else:
        print(f"Warning: Data selection not defined for {model_name}. Using processed (scaled) data by default.")
        current_X_train = X_train_processed
        current_X_test = X_test_processed

    # Train the model
    model.fit(current_X_train, y_train)

    # Make predictions
    y_pred = model.predict(current_X_test)
    try:
        y_pred_proba = model.predict_proba(current_X_test)[:, 1]
    except AttributeError:
        y_pred_proba = model.decision_function(current_X_test)
      
        print(f"Model {model_name} does not have predict_proba. ROC AUC might be based on decision_function or be inaccurate.")
        
        pass

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    roc_auc = np.nan
    try:
        y_pred_proba_for_roc = model.predict_proba(current_X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba_for_roc)
    except Exception as e:
        print(f"Could not compute ROC AUC for {model_name}: {e}")


    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc
    }

    print(f"Results for {model_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"  ROC AUC: {roc_auc:.4f}")
    else:
        print(f"  ROC AUC: Not computed")
        
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    print("-" * 40 + "\n")

print("\n--- All models trained and evaluated. ---")

# --- 9. Results Comparison ---
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='F1-Score', ascending=False)

print("\n--- Model Comparison ---")
print(results_df)

# Plotting the results
results_df_plot = results_df.copy()
if 'ROC AUC' in results_df_plot.columns:
    if results_df_plot['ROC AUC'].isnull().all():
        results_df_plot = results_df_plot.drop(columns=['ROC AUC'])

results_df_plot.plot(kind='bar', figsize=(18, 10)) 
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45, ha="right")
plt.legend(loc='best') 
plt.tight_layout()
plt.show()


print("\n--- Script Finished ---")
