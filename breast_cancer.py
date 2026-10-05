"""
Breast Cancer Diagnosis Prediction
==================================

Predicts whether a breast tumour is Malignant (M) or Benign (B) from 30
measurements of cell nuclei (Wisconsin Diagnostic Breast Cancer dataset),
and compares four models:

    * Logistic Regression  (main model)
    * K-Nearest Neighbors
    * Decision Tree
    * Linear Regression    (for comparison only, thresholded at 0.5)

Run:
    pip install -r requirements.txt
    python breast_cancer.py

Charts are saved to images/ and the metrics to results.json.
Educational project only - not a medical diagnostic tool.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files (works without a display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

BENIGN_COLOR = "#2a78d6"
MALIGNANT_COLOR = "#eb6834"
sns.set_theme(style="whitegrid")


def save(fig, name):
    fig.savefig(IMAGES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved images/{name}")


# ---------------------------------------------------------------------------
# 1. Load and clean the data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# 'id' is just a record number and 'Unnamed: 32' is an empty column in the Kaggle file
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

print("\nDiagnosis values:", df["diagnosis"].unique())
print("Missing values in total:", int(df.isnull().sum().sum()))

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
# Malignant (M) -> 1, Benign (B) -> 0
df["diagnosis"] = np.where(df["diagnosis"].astype(str).str.strip() == "M", 1, 0)

# X = feature matrix (tumour measurements), y = target (diagnosis)
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 80/20 split; random_state=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardise features: learn mean/spread on the training set only (prevents data leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------------------------
print("\nCreating charts...")
print(df.describe().T[["mean", "std", "min", "max"]].head(10))

# Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["diagnosis"].value_counts().sort_index()
bars = ax.bar(["Benign", "Malignant"], counts.values, color=[BENIGN_COLOR, MALIGNANT_COLOR], width=0.55)
ax.bar_label(bars, padding=3)
ax.set_title("Diagnosis count")
ax.set_ylabel("Number of tumours")
save(fig, "class_distribution.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Feature correlation heatmap")
save(fig, "correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------
results = {}


def evaluate(name, y_pred):
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc, 4))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# Logistic Regression (main model)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
evaluate("Logistic Regression", y_pred_log)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
evaluate("KNN", y_pred_knn)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluate("Decision Tree", y_pred_dt)

fig, ax = plt.subplots(figsize=(22, 11))
plot_tree(
    dt,
    filled=True,
    feature_names=X.columns.tolist(),
    class_names=["Benign", "Malignant"],
    rounded=True,
    fontsize=8,
    max_depth=3,
    ax=ax,
)
ax.set_title("Decision tree (first 3 levels)")
save(fig, "decision_tree.png")

# Linear Regression (for comparison - regression output thresholded at 0.5)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
y_pred_lin_class = (y_pred_lin > 0.5).astype(int)
evaluate("Linear Regression", y_pred_lin_class)
print("R2 score:", round(r2_score(y_test, y_pred_lin), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred_lin), 4))

# ---------------------------------------------------------------------------
# 5. Model comparison
# ---------------------------------------------------------------------------
print("\n=== Model comparison (test accuracy) ===")
for model, r in results.items():
    print(f"{model:<20} {r['accuracy']:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
names = list(results)
accs = [results[n]["accuracy"] * 100 for n in names]
bars = ax.barh(names[::-1], accs[::-1], color=BENIGN_COLOR, height=0.55)
ax.bar_label(bars, fmt="%.1f%%", padding=4)
ax.set_xlim(80, 100)
ax.set_xlabel("Test accuracy (%)")
ax.set_title("Model comparison")
save(fig, "model_comparison.png")

# Confusion matrix of the main model
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred_log),
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Logistic Regression - confusion matrix")
save(fig, "confusion_matrix.png")

# Most influential features in the logistic regression model
coef = pd.Series(log_model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False).head(10)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(
    coef.index[::-1],
    coef.values[::-1],
    color=[MALIGNANT_COLOR if v > 0 else BENIGN_COLOR for v in coef.values[::-1]],
    height=0.6,
)
ax.axvline(0, color="#888", lw=1)
ax.set_title("Top 10 features (logistic regression weights)")
ax.set_xlabel("Weight  (positive -> more likely malignant)")
save(fig, "feature_importance.png")

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
summary = {
    "dataset": {
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "benign": int((y == 0).sum()),
        "malignant": int((y == 1).sum()),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    },
    "models": results,
    "linear_regression_r2": round(r2_score(y_test, y_pred_lin), 4),
    "linear_regression_mse": round(mean_squared_error(y_test, y_pred_lin), 4),
    "top_features": {k: round(float(v), 4) for k, v in coef.items()},
}
(BASE_DIR / "results.json").write_text(json.dumps(summary, indent=2))
print("\nSaved results.json")
