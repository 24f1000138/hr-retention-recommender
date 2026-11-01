"""Re-scores the current workforce with the deployed ensemble and updates
results.json (band counts, department risk, action frequency, sample
recommendations) and figures/16_action_frequency.png.

Run after ensemble.py:  python rescore.py
"""
import json
import os
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from recommender import CATEGORICAL, ENGINEERED, NUMERIC, add_features, load_clean, recommend

HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid", palette="deep")
bundle = joblib.load(os.path.join(HERE, "model.joblib"))
with open(os.path.join(HERE, "results.json")) as fh:
    R = json.load(fh)

df = load_clean(os.path.join(HERE, "data", "hr_attrition.csv"))
current = df[df["Attrition"] == "No"].copy()
probs = bundle["ensemble"].predict_proba(add_features(current)[CATEGORICAL + NUMERIC + ENGINEERED])[:, 1]
recs = [recommend(r[CATEGORICAL + NUMERIC].to_dict(), bundle, p) for (_, r), p in zip(current.iterrows(), probs)]
current["Risk"] = [r["probability"] for r in recs]
current["Band"] = [r["band"] for r in recs]

R["deployed_model"] = bundle["ensemble_name"]
R["band_counts"] = current["Band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int).to_dict()
cnt = Counter(a["driver"] for r, b in zip(recs, current["Band"]) if b != "Low" for a in r["actions"])
R["action_frequency"] = [{"Driver": k, "Employees": v} for k, v in cnt.most_common(10)]
R["dept_risk"] = (current.groupby("Department")["Risk"].mean() * 100).round(1).to_dict()

R["sample_recommendations"] = []
for idx, r in current.sort_values("Risk", ascending=False).head(5).iterrows():
    rec = recs[current.index.get_loc(idx)]
    R["sample_recommendations"].append({
        "Employee": int(idx), "JobRole": r["JobRole"], "Age": int(r["Age"]),
        "Risk": round(rec["probability"] * 100, 1), "Band": rec["band"],
        "Segment": rec["segment_name"], "PayGap": round(rec["pay_gap_pct"], 1),
        "Actions": [a["action"] for a in rec["actions"][:3]],
        "Drivers": [a["driver"] for a in rec["actions"][:3]]})

plt.figure(figsize=(8, 4.5))
pd.Series({d["Driver"]: d["Employees"] for d in R["action_frequency"]}).sort_values().plot.barh(color="#4C72B0")
plt.title("Most common retention drivers (current medium/high-risk staff)")
plt.xlabel("Employees"); plt.ylabel("")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "figures", "16_action_frequency.png"), dpi=150); plt.close()

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(R, fh, indent=2, default=str)
print(json.dumps({k: R[k] for k in ["deployed_model", "band_counts", "dept_risk", "action_frequency"]}, indent=1))
