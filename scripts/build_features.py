import numpy as np
import pandas as pd
import os

DATA = "data"
os.makedirs("output", exist_ok=True)

#Load
companies = pd.read_csv(f"{DATA}/companies.csv", parse_dates=["signup_date"])
usage = pd.read_csv(f"{DATA}/usage_monthly.csv", parse_dates=["month"])
support = pd.read_csv(f"{DATA}/support_monthly.csv", parse_dates=["month"])
billing = pd.read_csv(f"{DATA}/billing_monthly.csv", parse_dates=["month"])

#Helper
def last_n_months(df, n=3):
    df = df.sort_values(["company_id", "month"])
    return df.groupby("company_id").tail(n)

WINDOW = 3
usage_recent = last_n_months(usage, WINDOW)
support_recent = last_n_months(support, WINDOW)
billing_recent = last_n_months(billing, WINDOW)

#Features
#Product
prod = usage_recent.groupby("company_id").agg(
    avg_active_users = ("active_users", "mean"),
    avg_logins = ("logins", "mean"),
    avg_features_used = ("features_used", "mean"),
    avg_key_actions = ("key_action_count", "mean")
).reset_index()

#Support
supp = support_recent.groupby("company_id").agg(
    avg_resolution_hours = ("avg_resolution_hours", "mean"),
    total_tickets = ("tickets_raised", "sum"),
    total_breached = ("tickets_breached_sla", "sum")
).reset_index()
supp["sla_breach_rate"] = np.where(
    supp["total_tickets"] > 0,
    supp["total_breached"] / supp["total_tickets"], 0)

#Billing
bill = billing_recent.groupby("company_id").agg(
    avg_payment_delay = ("payment_delay_days", "mean"),
    payment_failures = ("payment_failed", "sum")
).reset_index()

#Engagement
def usage_trend(g):
    g = g.sort_values("month")
    if len(g) < 2:
        return 0.0
    first = g["active_users"].iloc[0]
    last = g["active_users"].iloc[-1]
    if first == 0:
        return 0.0
    return (last - first) / first

trend = usage_recent.groupby("company_id").apply(usage_trend, include_groups=False).reset_index()
trend.columns = ["company_id", "usage_trend_pct"]

#Merge
feat = companies.copy()
feat = feat.merge(prod, on="company_id", how="left")
feat = feat.merge(supp, on="company_id", how="left")
feat = feat.merge(bill, on="company_id", how="left")
feat = feat.merge(trend, on="company_id", how="left")
feat = feat.fillna(0)

#seat utilization
feat["seat_utilization"] = np.where(
    feat["seats_licensed"] > 0,
    feat["avg_active_users"] / feat["seats_licensed"], 0)

#Normalise Dimensions
def clip01(x):
    return np.clip(x, 0, 1)

util_score = clip01(feat["seat_utilization"]) * 100
feature_score = clip01(feat["avg_features_used"] / 12) * 100
feat["product_health"] = 0.65 * util_score + 0.35 * feature_score

res_score = clip01((72 - feat["avg_resolution_hours"]) / (72-4)) * 100
breach_score = clip01(1 - feat["sla_breach_rate"]) * 100
feat["support_health"] = 0.5 * res_score + 0.5 * breach_score

delay_score = clip01((30 - feat["avg_payment_delay"]) / 30) * 100
failure_score = np.where(feat["payment_failures"] > 0, 40, 100)
feat["billing_health"] = 0.6 * delay_score + 0.4 * failure_score

feat["trend_health"] = clip01((feat["usage_trend_pct"] + 0.5) / 1.0) * 100

#final health score
W_PRODUCT, W_SUPPORT, W_BILLING, W_TREND = 0.40, 0.25, 0.20, 0.15
feat["health_score"] = (
    W_PRODUCT * feat["product_health"] +
    W_SUPPORT * feat["support_health"] +
    W_BILLING * feat["billing_health"] +
    W_TREND * feat["trend_health"]
).round(1)

#Segmentation
def segment(score):
    if score < 40:  return "At-Risk"
    elif score < 70: return "Stable"
    else:            return "Healthy"
feat["health_segment"] = feat["health_score"].apply(segment)

#Save
cols = [
    "company_id", "company_name", "industry", "company_size", "region",
    "plan_tier", "seats_licensed", "mrr", "signup_date",
    "seat_utilization", "avg_features_used", "avg_resolution_hours",
    "sla_breach_rate", "avg_payment_delay", "payment_failures",
    "usage_trend_pct",
    "product_health", "support_health", "billing_health", "trend_health",
    "health_score", "health_segment",
    "is_churned", "churn_month"
]
feat[cols].round(2).to_csv(f"{DATA}/features_health.csv", index=False)

#Validation
from scipy.stats import pointbiserialr
corr_pb, _ = pointbiserialr(feat["is_churned"], feat["health_score"])

print("\n Health Score Validation")
print(f"Rows in features_health.csv: {len(feat)}")
print(f"\nHealth score distribution:")
print(feat["health_score"].describe().round(1).to_string())
print(f"\nSegment counts:")
print(feat["health_segment"].value_counts().to_string())
print(f"\nAvg health score by churn status:")
print(feat.groupby("is_churned")["health_score"].mean().round(1).to_string())
print(f"\nChurn rate by health segment:")
seg = feat.groupby("health_segment")["is_churned"].agg(["mean", "count"])
seg.columns = ["churn_rate", "n_companies"]
print(seg.round(3).to_string())
print(f"\nPoint-biserial correlation between health score and churn: {corr_pb:.3f}")
print(f"(Expect NEGATIVE ~ -0.4 to -0.7)")
print(f"\nARR at risk (At-Risk segment MRR sum):"
      f"Rs.{feat[feat['health_segment']=='At-Risk']['mrr'].sum():,.0f}")

