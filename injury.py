# -------------------- 1. Imports --------------------
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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
X = df.drop(columns=["Athlete ID", "Date", "injury"])
y = df["injury"]

# -------------------- 3. Scaling --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- 4. Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------- 5. SMOTE Resampling --------------------
pos = sum(y_train == 1)
neg = sum(y_train == 0)
imbalance_ratio = neg / pos

smote = SMOTE(sampling_strategy=0.3, k_neighbors=3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------------------- 6. PCA for SVM only --------------------
pca = PCA(n_components=0.95, random_state=42)
X_train_svm = pca.fit_transform(X_train_res)
X_test_svm = pca.transform(X_test)

# -------------------- 7. Initialize Models --------------------
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
svm_model = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42)
xgb_model = XGBClassifier(
    use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=imbalance_ratio, random_state=42
)
lgb_model = LGBMClassifier(
    class_weight='balanced', random_state=42,
    metric='auc', n_estimators=200, learning_rate=0.05
)

models = {
    "Logistic Regression": log_reg,
    "Linear SVM (PCA)": svm_model,
    "XGBoost": xgb_model,
    "LightGBM": lgb_model
}

# -------------------- 8. Evaluation Function --------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, name, threshold=0.3):
    print(f"\n{'='*30}\nEvaluating: {name}")
    model.fit(X_train, y_train)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_pred = model.predict(X_test)
        y_proba = None
        auc = 'N/A'
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{name} - ROC Curve")
        plt.show()

    return {
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1 Score": f1, "ROC AUC": auc
    }

# -------------------- 9. Train and Evaluate Models --------------------
results = []
results.append(evaluate_model(log_reg, X_train_res, y_train_res, X_test, y_test, "Logistic Regression"))
results.append(evaluate_model(svm_model, X_train_svm, y_train_res, X_test_svm, y_test, "Linear SVM (PCA)", threshold=0.5))
results.append(evaluate_model(xgb_model, X_train_res, y_train_res, X_test, y_test, "XGBoost", threshold=0.3))
results.append(evaluate_model(lgb_model, X_train_res, y_train_res, X_test, y_test, "LightGBM", threshold=0.3))

# -------------------- 10. Feature Importance (XGBoost) --------------------
xgb_model.fit(X_train_res, y_train_res)
importances = xgb_model.feature_importances_
feature_names = X.columns
imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
imp_df = imp_df.sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=imp_df, palette="mako")
plt.title("Top 15 Important Features (XGBoost)")
plt.tight_layout()
plt.show()

# -------------------- 11. Model Comparison --------------------
comp_df = pd.DataFrame(results)
print("\nModel Comparison:\n", comp_df)

plt.figure(figsize=(10, 6))
sns.barplot(x="F1 Score", y="Model", data=comp_df.sort_values("F1 Score"), palette="coolwarm")
plt.title("Model Comparison - F1 Score")
plt.tight_layout()
plt.show()
