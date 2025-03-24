import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from google.colab import files
# -------------------- 2. Load Data --------------------
print("Please upload your CSV file")
uploaded = files.upload()

# Extract file name
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# Display dataset information
print("Dataset Overview:\n", df.head())
print("\nDataset Info:\n")
df.info()

print("Shape of dataset:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Class distribution:\n", df['injury'].value_counts())

# -------------------- 3. Feature Selection --------------------
X = df.drop(columns=["Athlete ID", "Date", "injury"])
y = df["injury"]

# -------------------- 4. Scaling --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- 5. Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------- 6. Apply SMOTE --------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE Resampling:")
print(pd.Series(y_train_res).value_counts())

# -------------------- 7. PCA for SVM only --------------------
pca = PCA(n_components=0.95, random_state=42)
X_train_svm = pca.fit_transform(X_train_res)
X_test_svm = pca.transform(X_test)

print("Original feature count:", X_train_res.shape[1])
print("Reduced feature count after PCA:", X_train_svm.shape[1])

# -------------------- 8. Initialize Models --------------------
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
svm_model = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1, random_state=42)

# -------------------- 9. Evaluation Function --------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, use_proba=True):
    print(f"\n{'='*30}\nEvaluating: {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if use_proba and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        auc = 'N/A'

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{model_name} - ROC Curve")
        plt.show()

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": auc
    }

# -------------------- 10. Train Models --------------------
results = []

# Logistic Regression
results.append(evaluate_model(log_reg, X_train_res, y_train_res, X_test, y_test, "Logistic Regression"))

# Linear SVM with PCA
results.append(evaluate_model(svm_model, X_train_svm, y_train_res, X_test_svm, y_test, "Linear SVM (PCA)", use_proba=False))

# XGBoost
results.append(evaluate_model(xgb_model, X_train_res, y_train_res, X_test, y_test, "XGBoost"))

# -------------------- 11. XGBoost Feature Importance --------------------
xgb_model.fit(X_train_res, y_train_res)
importances = xgb_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df, palette="viridis")
plt.title("Top 15 Important Features (XGBoost)")
plt.tight_layout()
plt.show()

# -------------------- 12. Model Comparison --------------------
comparison_df = pd.DataFrame(results)
print("\nModel Comparison:\n", comparison_df)

plt.figure(figsize=(10, 6))
sns.barplot(x="F1 Score", y="Model", data=comparison_df.sort_values("F1 Score", ascending=True), palette="coolwarm")
plt.title("Model Comparison - F1 Score")
plt.xlabel("F1 Score")
plt.tight_layout()
plt.show()


