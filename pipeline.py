"""
Heart Disease Prediction - Full Pipeline
Stages: Data Cleaning -> EDA -> Feature Engineering -> Model Building -> Evaluation
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report, roc_curve)

sns.set_theme(style="whitegrid")
PALETTE = ["#990011", "#2F3C7E", "#A26769"]

RAW_PATH = "../data/raw_heart_patients.csv"
CLEAN_PATH = "../data/cleaned_heart_patients.csv"
VIS_DIR = "../visuals"
MODEL_DIR = "../models"
import os
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)

# ============================================================
# STAGE 3: DATA CLEANING & PREPROCESSING
# ============================================================
df = pd.read_csv(RAW_PATH)
print("Raw shape:", df.shape)

# --- Drop Patient_ID (identifier, not predictive) ---
df = df.drop(columns=["Patient_ID"])

# --- Fix Age: contains 'unknown' strings mixed with numeric strings ---
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")  # 'unknown' -> NaN

# --- Standardize Gender casing ---
df["Gender"] = df["Gender"].str.strip().str.lower().map({"male": "Male", "female": "Female"})

# --- Standardize Exercise_Induced_Angina casing ---
df["Exercise_Induced_Angina"] = df["Exercise_Induced_Angina"].str.strip().str.lower().map({"yes": "Yes", "no": "No"})

# --- Standardize Chest_Pain_Type casing (already consistent, just strip) ---
df["Chest_Pain_Type"] = df["Chest_Pain_Type"].str.strip()

# --- Standardize target Heart_Disease: 'zero'/'0' -> 0, 'one'/'1' -> 1 ---
target_map = {"zero": 0, "one": 1, "0": 0, "1": 1, 0: 0, 1: 1}
df["Heart_Disease"] = df["Heart_Disease"].map(target_map)

# --- Drop rows where target is missing (can't train/validate without label) ---
before = len(df)
df = df.dropna(subset=["Heart_Disease"])
print(f"Dropped {before - len(df)} rows with missing target")

# --- Drop exact duplicate rows ---
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} exact duplicate rows")

df["Heart_Disease"] = df["Heart_Disease"].astype(int)

# --- Handle remaining missing values ---
num_cols = ["Age", "Resting_BP_mmHg", "Cholesterol_mg/dl", "Max_Heart_Rate", "ST_Depression"]
cat_cols = ["Gender", "Chest_Pain_Type", "Exercise_Induced_Angina"]

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# --- Handle outliers via IQR capping (clinical data - cap not remove, to preserve extreme-but-real cases) ---
for c in num_cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[c] = df[c].clip(lower, upper)

print("Cleaned shape:", df.shape)
print(df.isna().sum())
df.to_csv(CLEAN_PATH, index=False)

# ============================================================
# STAGE 4: EXPLORATORY DATA ANALYSIS
# ============================================================

# Target balance
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Heart_Disease", hue="Heart_Disease", data=df, palette=PALETTE[:2], legend=False)
ax.set_xticklabels(["No Disease", "Heart Disease"])
plt.title("Target Class Balance")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_target_balance.png", dpi=150)
plt.close()

# Age distribution by target
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="Age", hue="Heart_Disease", kde=True, palette=PALETTE[:2], element="step")
plt.title("Age Distribution by Heart Disease Status")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_age_distribution.png", dpi=150)
plt.close()

# Correlation heatmap (numeric only)
plt.figure(figsize=(7, 6))
corr = df[num_cols + ["Heart_Disease"]].corr()
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()

# Boxplots of key numeric features by target
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.flat, num_cols):
    sns.boxplot(x="Heart_Disease", y=c, hue="Heart_Disease", data=df, ax=ax, palette=PALETTE[:2], legend=False)
    ax.set_title(c)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/04_boxplots_by_target.png", dpi=150)
plt.close()

# Chest pain type vs disease
plt.figure(figsize=(7, 4))
sns.countplot(x="Chest_Pain_Type", hue="Heart_Disease", data=df, palette=PALETTE[:2])
plt.xticks(rotation=20)
plt.title("Chest Pain Type vs Heart Disease")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/05_chestpain_vs_target.png", dpi=150)
plt.close()

# Gender vs disease
plt.figure(figsize=(5, 4))
sns.countplot(x="Gender", hue="Heart_Disease", data=df, palette=PALETTE[:2])
plt.title("Gender vs Heart Disease")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/06_gender_vs_target.png", dpi=150)
plt.close()

print("EDA plots saved.")

# ============================================================
# STAGE 5: FEATURE ENGINEERING
# ============================================================
df_fe = df.copy()

# One-hot encode categorical variables
df_fe = pd.get_dummies(df_fe, columns=["Gender", "Chest_Pain_Type", "Exercise_Induced_Angina"], drop_first=True)

# New engineered feature: Age group (bins)
df_fe["Age_Group"] = pd.cut(df["Age"], bins=[0, 40, 55, 70, 120], labels=["<40", "40-55", "56-70", "70+"])
df_fe = pd.get_dummies(df_fe, columns=["Age_Group"], drop_first=True)

# New engineered feature: Pulse Pressure proxy not available (no systolic/diastolic split) -> skip
# New engineered feature: High cholesterol flag (clinical threshold 240 mg/dl)
df_fe["High_Cholesterol_Flag"] = (df["Cholesterol_mg/dl"] > 240).astype(int)

feature_cols = [c for c in df_fe.columns if c != "Heart_Disease"]
X = df_fe[feature_cols]
y = df_fe["Heart_Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
num_to_scale = ["Age", "Resting_BP_mmHg", "Cholesterol_mg/dl", "Max_Heart_Rate", "ST_Depression"]
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_to_scale] = scaler.fit_transform(X_train[num_to_scale])
X_test_scaled[num_to_scale] = scaler.transform(X_test[num_to_scale])

joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_columns.joblib")

print("Feature-engineered shape:", X.shape)

# ============================================================
# STAGE 6: MODEL BUILDING
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

# Hyperparameter tuning for Random Forest (best candidate)
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring="f1", n_jobs=1)
rf_grid.fit(X_train_scaled, y_train)
models["Random Forest (Tuned)"] = rf_grid.best_estimator_
print("Best RF params:", rf_grid.best_params_)

# ============================================================
# STAGE 7: MODEL EVALUATION
# ============================================================
results = []
roc_data = {}
for name, model in models.items():
    if name != "Random Forest (Tuned)":
        model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc})
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)

# --- Select the ACTUAL best model by F1-score (not assumed in advance) ---
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
print(f"\nBest model selected: {best_name}")

y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu_r", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/07_confusion_matrix.png", dpi=150)
plt.close()

print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, y_pred_best))

joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib")
with open(f"{MODEL_DIR}/best_model_name.txt", "w") as f:
    f.write(best_name)

# Feature importance (only for tree-based models)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
else:
    rf_fallback = models["Random Forest"]
    importances = pd.Series(rf_fallback.feature_importances_, index=feature_cols).sort_values(ascending=False)

plt.figure(figsize=(7, 6))
sns.barplot(x=importances.values[:10], y=importances.index[:10], hue=importances.index[:10], palette="RdBu_r", legend=False)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/08_feature_importance.png", dpi=150)
plt.close()
importances.to_csv(f"{MODEL_DIR}/feature_importances.csv")

# ROC curve comparison plot
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/09_roc_curves.png", dpi=150)
plt.close()

# Save summary stats used in report/slides
summary = {
    "n_rows_raw": 1020,
    "n_rows_cleaned": int(df.shape[0]),
    "n_features_final": int(X.shape[1]),
    "best_model": best_name,
    "rf_tuned_best_params": rf_grid.best_params_,
    "test_accuracy": float(results_df.iloc[0]["Accuracy"]),
    "test_precision": float(results_df.iloc[0]["Precision"]),
    "test_recall": float(results_df.iloc[0]["Recall"]),
    "test_f1": float(results_df.iloc[0]["F1-Score"]),
    "test_auc": float(results_df.iloc[0]["ROC-AUC"]),
    "top_features": importances.index[:5].tolist(),
}
with open(f"{MODEL_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nPipeline complete. Summary:")
print(json.dumps(summary, indent=2))
