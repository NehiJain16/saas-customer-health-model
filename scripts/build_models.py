import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report, accuracy_score)

DATA = "data"
os.makedirs("output", exist_ok=True)

#Load
df = pd.read_csv(f"{DATA}/features_health.csv")

#Churn Model
features = [
    "seat_utilization",
    "avg_resolution_hours",
    "avg_payment_delay",
]
X = df[features].copy()
y = df["is_churned"].copy()

#scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000, class_weight="balanced",
                           C=0.5)
model.fit(X_train, y_train)

#metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

#churn probability
df["churn_probability"] = model.predict_proba(X_scaled)[:, 1].round(3)

#feature importance
importance = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_[0].round(3),
    "abs_importance": np.abs(model.coef_[0]).round(3)
}).sort_values("abs_importance", ascending=False)
importance["direction"] = np.where(importance["coefficient"] > 0,
                                   "increases churn", "decreases churn")
importance.to_csv(f"output/churn_feature_importance.csv", index=False)

#Expansion Propensity
def clip01(x):
    return np.clip(x, 0, 1)

#components scores
util_comp = clip01((df["seat_utilization"] - 0.5) / 0.5)
health_comp = clip01(df["health_score"] / 100)
trend_comp = clip01((df["usage_trend_pct"] + 0.2) / 0.4)
billing_comp = np.where(df["payment_failures"] == 0,
                        clip01((30 - df["avg_payment_delay"]) / 30), 0.3)
lowrisk_comp = clip01(1 - df["churn_probability"])

df["expansion_score"] = (
    0.30 * util_comp +
    0.25 * health_comp +
    0.20 * trend_comp +
    0.10 * billing_comp +
    0.15 * lowrisk_comp
) * 100
df["expansion_score"] = df["expansion_score"].round(1)

#Flag
df["expansion_flag"] = np.where(
    (df["expansion_score"] >= 65) & (df["is_churned"] == 0),
    "Expansion-Ready", "Not Ready")

#Action Quadrant
def quadrant(row):
    high_health = row["health_score"] >= 65
    high_risk = row["churn_probability"] >= 0.4
    if high_health and not high_risk: return "EXPAND"
    if high_health and high_risk: return "NURTURE"
    if not high_health and not high_risk: return "MONITOR"
    return "RESCUE"
df["action_quadrant"] = df.apply(quadrant, axis=1)

#Save Final
df.to_csv(f"{DATA}/features_final.csv", index=False)

#Save Metrics File
with open("output/churn_model_metrics.txt", "w") as f:
    f.write("Churn Model - Logistic Regression (interpretable)\n")
    f.write("="*55 + "\n")
    f.write(f"Test Accuracy: {acc:.3f}\n")
    f.write(f"ROC-AUC: {auc:.3f}\n\n")
    f.write("Confusion Matrix (rows=actual, columns=predicted):\n")
    f.write(f"{cm}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("\nFeature Importance (standardized coefficients):\n")
    f.write(importance.to_string(index=False))

#Validation printout
print("\nChurn Model")
print(f"Test Accuracy: {acc:.3f}")
print(f"ROC-AUC: {auc:.3f}  (good if 0.75+)")
print(f"\nConfusion Matrix (rows=actual 0/1, columns=predicted 0/1):\n{cm}")
print(f"\nTop churn drivers (by |coefficient|):")
print(importance.to_string(index=False))

print("\nExpansion Propensity")
print(f"Expansion-Ready companies: {(df['expansion_flag']== 'Expansion-Ready').sum()}")
print(f"Expansion score distribution:")
print(df["expansion_score"].describe().round(1).to_string())

print(f"\nAction Quadrant (2x2):")
q = df.groupby("action_quadrant").agg(
    n_companies=("company_id", "count"),
    total_mrr=("mrr", "sum"),
    churn_rate=("is_churned", "mean")
).round(2)
print(q.to_string())

print("\nSanity check: avg churn_probability by health_segment")
print(df.groupby("health_segment")["churn_probability"].mean().round(3).to_string())

print("\nExpansion-Ready: total MRR opportunity")
exp_mrr = df[df["expansion_flag"] == "Expansion-Ready"]["mrr"].sum()
print(f"Rs.{exp_mrr:,.0f} of MRR sits in expansion-ready accounts")