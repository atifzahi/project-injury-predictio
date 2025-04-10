import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# Load dataset
from google.colab import files
import io
print("Please upload your dataset CSV file")
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded[list(uploaded.keys())[0]]))
df.dropna(inplace=True)

# Define features and target
X = df.drop(columns=['Athlete ID', 'Date', 'injury'])
y = df['injury']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Undersample training data to balance class ratio
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df['injury'] = y_train.reset_index(drop=True)
minor = train_df[train_df['injury'] == 1]
major = train_df[train_df['injury'] == 0].sample(n=len(minor)*6, random_state=42)
balanced_train = pd.concat([minor, major])
X_train_bal = balanced_train.drop(columns='injury')
y_train_bal = balanced_train['injury']

# Define models
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=6, random_state=42),
    "LightGBM": LGBMClassifier(class_weight='balanced', n_estimators=300, learning_rate=0.03, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, random_state=42),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning
grid_params_logreg = {'C': [0.01, 0.1, 1, 10]}
grid_params_et = {'n_estimators': [100, 200], 'max_depth': [5, 10]}

for name, model in models.items():
    print(f"\nTraining model: {name}")

    if name == "Logistic Regression":
        model = GridSearchCV(model, grid_params_logreg, scoring='recall', cv=cv)
    elif name == "Extra Trees":
        model = GridSearchCV(model, grid_params_et, scoring='recall', cv=cv)

    model.fit(X_train_bal, y_train_bal)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_scaled)
    if not hasattr(model, 'predict_proba'):
        y_proba = 1 / (1 + np.exp(-y_proba))

    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1, best_thresh = 0, 0.5
    for thresh in thresholds:
        y_pred_temp = (y_proba >= thresh).astype(int)
        f1_temp = f1_score(y_test, y_pred_temp)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_thresh = thresh

    y_pred = (y_proba >= best_thresh).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"{name} - ROC Curve")
    plt.show()

    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title(f"{name} - Precision-Recall Curve")
    plt.show()

    print(f"Best threshold = {best_thresh:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    f1_scores = []
    for train_idx, val_idx in cv.split(X_train_bal, y_train_bal):
        X_tr, X_val = X_train_bal.iloc[train_idx], X_train_bal.iloc[val_idx]
        y_tr, y_val = y_train_bal.iloc[train_idx], y_train_bal.iloc[val_idx]
        model.fit(X_tr, y_tr)
        val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_val)
        val_pred = (val_proba >= best_thresh).astype(int)
        f1_scores.append(f1_score(y_val, val_pred))

    avg_f1_cv = np.mean(f1_scores)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": auc,
        "Best Threshold": best_thresh,
        "F1 CV Avg": avg_f1_cv
    })

# Final results
result_df = pd.DataFrame(results)
print("\nModel Comparison:\n", result_df)
sns.barplot(data=result_df, x="F1 Score", y="Model", palette="coolwarm")
plt.title("Model Comparison - F1 Score")
plt.tight_layout()
plt.show()
