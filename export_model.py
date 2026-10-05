"""Export the trained logistic regression model + test samples for the website demo (website/model.json)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "data" / "data.csv").drop(columns=["id", "Unnamed: 32"], errors="ignore")
df["diagnosis"] = np.where(df["diagnosis"].astype(str).str.strip() == "M", 1, 0)
X, y = df.drop("diagnosis", axis=1), df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

out = {
    "features": X.columns.tolist(),
    "mean": scaler.mean_.round(6).tolist(),
    "scale": scaler.scale_.round(6).tolist(),
    "coef": model.coef_[0].round(6).tolist(),
    "intercept": round(float(model.intercept_[0]), 6),
    "min": X.min().round(4).tolist(),
    "max": X.max().round(4).tolist(),
    "samples": [{"x": X_test.iloc[i].round(5).tolist(), "y": int(y_test.iloc[i])} for i in range(len(X_test))],
}
(BASE / "model.json").write_text(json.dumps(out, separators=(",", ":")))
# sanity check: JS-style prediction matches sklearn
z = (X_test.values - scaler.mean_) / scaler.scale_ @ np.eye(30)
p = 1 / (1 + np.exp(-(z @ model.coef_[0] + model.intercept_[0])))
assert ((p > 0.5).astype(int) == model.predict(scaler.transform(X_test))).all()
print("model.json written,", len(out["samples"]), "samples; manual prediction matches sklearn")
