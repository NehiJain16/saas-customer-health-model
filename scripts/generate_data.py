import numpy as np
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

#CONFIG
SEED = 42
N_COMPANIES = 500
N_MONTHS = 24
np.random.seed(SEED)

#Output folder
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

#Reference start month = 24 months ago from a fixed anchor
ANCHOR = datetime(2025, 1, 1)
START_MONTH = ANCHOR - relativedelta(months=N_MONTHS - 1)
MONTHS = [START_MONTH + relativedelta(months=i) for i in range(N_MONTHS)]

INDUSTRIES = ["Retail", "Logistics", "EdTech", "Healthcare",
              "Manufacturing", "IT Services", "BFSI"]
SIZES = ["SMB", "Mid", "Enterprise"]
REGIONS = ["North", "South", "East", "West"]
TIERS = ["Basic", "Pro", "Enterprise"]

NAME_PREFIX = ["Acme", "Zenith", "Apex", "Nova", "Orbit", "Vertex", "Pulse",
               "Quanta", "Stellar", "Bright", "Indus", "Bharat", "Lotus",
               "Sankalp", "Vega", "Onyx", "Prime", "Crest", "Echo0", "Drift"]
NAME_SUFFIX = ["Technologies", "Solutions", "Logistics", "Retail", "Labs",
               "Systems", "Industries", "Pvt Ltd", "Enterprises", "Networks"]

# 1. Generate Companies

def make_companies():
    rows = []
    for i in range(1, N_COMPANIES + 1):
        cid = f"CUST_{i:04d}"
        size = np.random.choice(SIZES, p=[0.5, 0.35, 0.15])
        #seats and MRR scale with size
        if size == "SMB":
            seats = np.random.randint(10, 51)
            tier = np.random.choice(TIERS, p=[0.6, 0.35, 0.05])
        elif size == "Mid":
            seats = np.random.randint(51, 201)
            tier = np.random.choice(TIERS, p=[0.2, 0.55, 0.25])
        else:
            seats = np.random.randint(201, 501)
            tier = np.random.choice(TIERS, p=[0.05, 0.35, 0.60])

        #MRR per seat varies by tier
        per_seat = {"Basic": 300, "Pro": 600, "Enterprise": 1200} [tier]
        mrr = int(seats * per_seat * np.random.uniform(0.85, 1.15))

        #signup date
        signup_offset = np.random.randint(0, N_MONTHS)
        signup_date = MONTHS[signup_offset]

        name = f"{np.random.choice(NAME_PREFIX)} {np.random.choice(NAME_SUFFIX)}"

        rows.append({
            "company_id": cid,
            "company_name": name,
            "industry": np.random.choice(INDUSTRIES),
            "company_size": size,
            "region": np.random.choice(REGIONS),
            "plan_tier": tier,
            "seats_licensed": seats,
            "mrr": mrr,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "signup_offset": signup_offset,
            "is_churned": 0,
            "churn_month": None
        })
    return pd.DataFrame(rows)


# 2. Assign a "Health Archetype"
def assign_archetypes(companies):
    archetypes = []
    for _, c in companies.iterrows():
        size = c["company_size"]
        tier = c["plan_tier"]

        #base probabilities 
        if size == "Enterprise":
            p = [0.72, 0.20, 0.08]
        elif size == "Mid":
            p = [0.58, 0.27, 0.15]
        else: #SMB
            p = [0.42, 0.30, 0.28]

        #tier nudges health up/down
        if tier == "Enterprise":
            p = [p[0] + 0.10, p[1], max (0.02, p[2] - 0.10)]
        elif tier == "Basic":
            p = [max(0.05, p[0] - 0.10), p[1], p[2] + 0.10]

        #normalize to sum to 1
        s = sum(p)
        p = [x / s for x in p]

        archetypes.append(np.random.choice(
            ["healthy", "declining", "problematic"], p=p))
        
    companies["archetype"] = archetypes
    return companies

# 3. Generate Monthly Usage / Support / Billing + churn logic

def make_monthly(companies):
    usage_rows, support_rows, billing_rows = [], [], []

    for _, c in companies.iterrows():
        cid = c["company_id"]
        seats = c["seats_licensed"]
        mrr = c["mrr"]
        arch = c["archetype"]
        start_idx = c["signup_offset"]

        #base health level by archetype
        if arch == "healthy":
            base_engage = np.random.uniform(0.7, 0.95)
            base_res = np.random.uniform(4, 18)
            base_delay = np.random.uniform(0, 5)
            decline = np.random.uniform(-0.005, 0.005)
        elif arch == "declining":
            base_engage = np.random.uniform(0.5, 0.75)
            base_res = np.random.uniform(12, 36)
            base_delay = np.random.uniform(3, 12)
            decline = np.random.uniform(-0.03, -0.01)
        else:
            base_engage = np.random.uniform(0.25, 0.5)
            base_res = np.random.uniform(36, 90)
            base_delay = np.random.uniform(8, 25)
            decline = np.random.uniform(-0.05, -0.02)

        churned = False
        churn_month_str = None

        for m_idx in range(start_idx, N_MONTHS):
            month = MONTHS[m_idx]
            months_since_signup = m_idx - start_idx

            if churned:
                break 

            # engagement ratio drifts over time
            engage = base_engage + decline * months_since_signup
            engage += np.random.normal(0, 0.05) 
            engage = float(np.clip(engage, 0.02, 1.0))

            active_users = int(np.clip(round(seats * engage), 0, seats))
            logins = int(max(0, active_users * np.random.uniform(8, 25)))
            features_used = int(np.clip(round(12 * engage + np.random.normal(0, 1)), 0, 12))
            key_action_count = int(max(0, active_users * np.random.uniform(2, 10)))
            
            usage_rows.append({
                "company_id": cid, "month": month.strftime("%Y-%m-%d"),
                "active_users": active_users, "logins": logins,
                "features_used": features_used, "key_action_count": key_action_count
            })

            # support
            tickets = int(np.clip(np.random.poisson(3 + (1 - engage) * 8), 0, 25))
            res_hours = float(max(2, base_res + np.random.normal(8, 6)))
            breached = int(np.clip(round(tickets * np.clip((res_hours - 24) / 72, 0, 1)
                                        + np.random.uniform(0, 1)), 0, tickets))
            support_rows.append({
                "company_id": cid, "month": month.strftime("%Y-%m-%d"),
                "tickets_raised": tickets,
                "avg_resolution_hours": round(res_hours, 1),
                "tickets_breached_sla": breached
            })
            
            # billing
            delay = int(max(0, base_delay + np.random.normal(0, 4)))
            failed = 1 if (delay > 20 and np.random.rand() < 0.4) else 0
            billing_rows.append({
                "company_id": cid, "month": month.strftime("%Y-%m-%d"),
                "invoice_amount": mrr,
                "payment_delay_days": delay,
                "payment_failed": failed
            })

            # CHURN LOGIC: probability rises as health worsens
            churn_pressure = (1 - engage) * 0.05 \
                            + (res_hours > 48) * 0.02 \
                            + (delay > 15) * 0.02 \
                            + (arch == "problematic") * 0.02
            # only allow churn after at least 2 months tenure
            if months_since_signup >= 2 and np.random.rand() < churn_pressure:
                churned = True 
                churn_month_str = month.strftime("%Y-%m-%d")
        
        # write churn back to companies
        companies.loc[companies["company_id"] ==cid, "is_churned"] = int(churned)
        companies.loc[companies["company_id"] ==cid, "churn_month"] = churn_month_str

    usage = pd.DataFrame(usage_rows)
    support = pd.DataFrame(support_rows) 
    billing = pd.DataFrame(billing_rows) 
    return companies, usage, support, billing

# Main
if __name__ == "__main__":
    print("Generating companies...")
    companies = make_companies()
    companies = assign_archetypes(companies)

    print("Generating monthly usage / support / billing + churn...")
    companies, usage, support, billing = make_monthly(companies)

    # drop helper columns before saving 
    companies_out = companies.drop(columns=["signup_offset", "archetype"])

    companies_out.to_csv(f"{OUT_DIR}/companies.csv", index=False)
    usage.to_csv(f"{OUT_DIR}/usage_monthly.csv", index=False)
    support.to_csv(f"{OUT_DIR}/support_monthly.csv", index=False) 
    billing.to_csv(f"{OUT_DIR}/billing_monthly.csv", index=False)

    #Validation Summary
    print("\nVALIDATION SUMMARY")
    print(f"Companies rows : {len(companies_out)}")
    print(f"Usage rows:{len(usage)}") 
    print(f"Support rows : {len(support)}")
    print(f"Billing rows : {len(billing)}")
    print(f"Overall churn rate : {companies_out['is_churned'].mean():.1%}")
    print(f"Avg MRR (active+churned) : Rs.{companies_out['mrr'].mean():,.0f}")
    print("\nChurn rate by plan tier:")
    print(companies_out.groupby('plan_tier')['is_churned'].mean().round(3).to_string()) 
    print("\nChurn rate by company size:")
    print(companies_out.groupby('company_size')['is_churned'].mean().round(3).to_string())
    print("\nSample company record:") 
    print(companies_out.iloc[0].to_string()) 
    print("\nDONE. 4 CSVs written to the data/ folder.")

