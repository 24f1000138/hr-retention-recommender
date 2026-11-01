"""Shared recommendation logic for the Employee Retention Recommender.

Used by analysis.py (to build and save the models) and app.py (to score
an employee and recommend retention actions).
"""
import numpy as np
import pandas as pd

TARGET = "Attrition"
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

CATEGORICAL = ["BusinessTravel", "Department", "EducationField", "Gender",
               "JobRole", "MaritalStatus", "OverTime"]

NUMERIC = ["Age", "DailyRate", "DistanceFromHome", "Education",
           "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
           "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
           "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
           "RelationshipSatisfaction", "StockOptionLevel",
           "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
           "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
           "YearsWithCurrManager"]

# Predictors for the salary benchmark (multiple linear regression).
INCOME_PREDICTORS = ["JobLevel", "TotalWorkingYears", "YearsAtCompany",
                     "Age", "PerformanceRating"]

# Features used for K-means segmentation.
CLUSTER_FEATURES = ["Age", "MonthlyIncome", "TotalWorkingYears",
                    "YearsAtCompany", "YearsSinceLastPromotion",
                    "JobSatisfaction", "EnvironmentSatisfaction",
                    "WorkLifeBalance", "OverTimeFlag"]

# Bands are set around the 16% base attrition rate; the ROC-optimal
# (Youden) threshold of about 0.28 falls inside the Medium band.
RISK_BANDS = [(0.0, 0.20, "Low"), (0.20, 0.40, "Medium"), (0.40, 1.01, "High")]


def risk_band(p):
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "High"


def load_clean(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop_duplicates()
    df["AttritionFlag"] = (df[TARGET] == "Yes").astype(int)
    df["OverTimeFlag"] = (df["OverTime"] == "Yes").astype(int)
    return df


# Ratios and indices used by the ensemble model (feature engineering)
ENGINEERED = ["IncomePerLevel", "TenureRatio", "PromotionGapRatio", "RoleTenureRatio",
              "ManagerTenureRatio", "JobHopRate", "SatisfactionIndex"]


def add_features(df):
    df = df.copy()
    df["IncomePerLevel"] = df["MonthlyIncome"] / df["JobLevel"]
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["RoleTenureRatio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
    df["ManagerTenureRatio"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
    df["JobHopRate"] = df["NumCompaniesWorked"] / (df["TotalWorkingYears"] + 1)
    df["SatisfactionIndex"] = df[["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
                                  "JobInvolvement", "RelationshipSatisfaction"]].mean(axis=1)
    return df


# Each rule: (driver name, model features it relates to, test on the
# employee record, recommended action, owner, priority weight)
RULES = [
    ("Regular overtime", ["OverTime"],
     lambda e, ctx: e["OverTime"] == "Yes",
     "Rebalance workload and cap overtime hours; review team staffing levels.",
     "Line manager", 3),
    ("Low absolute pay", ["MonthlyIncome"],
     lambda e, ctx: e["MonthlyIncome"] < 3000,
     "Review pay against the market for the role; consider a targeted "
     "increment or allowance.", "HR / Compensation", 2),
    ("Paid below the salary benchmark", ["MonthlyIncome", "JobLevel"],
     lambda e, ctx: ctx.get("pay_gap_pct", 0) <= -15,
     "Check pay equity: monthly income is more than 15% below the "
     "regression benchmark for this level and experience.",
     "HR / Compensation", 1),
    ("Low job satisfaction", ["JobSatisfaction"],
     lambda e, ctx: e["JobSatisfaction"] <= 2,
     "Hold a stay interview and enrich the role (new responsibilities, "
     "project rotation).", "Line manager", 2),
    ("Low environment satisfaction", ["EnvironmentSatisfaction"],
     lambda e, ctx: e["EnvironmentSatisfaction"] <= 2,
     "Address workplace concerns raised in surveys (tools, team climate, "
     "workspace).", "HR business partner", 2),
    ("Poor work-life balance", ["WorkLifeBalance"],
     lambda e, ctx: e["WorkLifeBalance"] <= 2,
     "Offer flexible hours or hybrid working days.", "Line manager", 2),
    ("Low job involvement", ["JobInvolvement"],
     lambda e, ctx: e["JobInvolvement"] <= 2,
     "Set clear goals and involve the employee in decision-making and "
     "recognition programmes.", "Line manager", 2),
    ("No stock options", ["StockOptionLevel"],
     lambda e, ctx: e["StockOptionLevel"] == 0,
     "Consider including the employee in the stock option / long-term "
     "incentive plan.", "HR / Compensation", 2),
    ("Overdue promotion", ["YearsSinceLastPromotion"],
     lambda e, ctx: e["YearsSinceLastPromotion"] >= 4,
     "Discuss a career path and review promotion readiness.",
     "HR / Talent", 2),
    ("Frequent business travel", ["BusinessTravel"],
     lambda e, ctx: e["BusinessTravel"] == "Travel_Frequently",
     "Reduce travel load or offer travel compensation and recovery days.",
     "Line manager", 2),
    ("Early-career, single, entry level", ["MaritalStatus", "JobLevel", "Age"],
     lambda e, ctx: e["MaritalStatus"] == "Single" and e["JobLevel"] == 1,
     "Assign a mentor and a 12-month development plan with visible growth "
     "milestones.", "HR / Talent", 1),
    ("Long commute", ["DistanceFromHome"],
     lambda e, ctx: e["DistanceFromHome"] >= 15,
     "Offer remote days or a transport allowance.", "HR", 1),
    ("Little recent training", ["TrainingTimesLastYear"],
     lambda e, ctx: e["TrainingTimesLastYear"] <= 1,
     "Enrol in skill-development or certification programmes.",
     "HR / L&D", 1),
    ("Low relationship satisfaction", ["RelationshipSatisfaction"],
     lambda e, ctx: e["RelationshipSatisfaction"] <= 1,
     "Facilitate team-building and manager coaching conversations.",
     "Line manager", 1),
    ("Many previous employers", ["NumCompaniesWorked"],
     lambda e, ctx: e["NumCompaniesWorked"] >= 5,
     "Agree a retention bonus or milestone-linked incentive.",
     "HR / Compensation", 1),
]


def feature_contributions(clf_pipeline, row_df, train_mean):
    """Per-feature contribution to the logit for one employee:
    coef * (x - training mean) in the transformed space, summed back to the
    original column names."""
    pre = clf_pipeline.named_steps["prep"]
    lr = clf_pipeline.named_steps["clf"]
    x = pre.transform(row_df)
    x = np.asarray(x.todense() if hasattr(x, "todense") else x)[0]
    contrib = lr.coef_[0] * (x - train_mean)
    names = pre.get_feature_names_out()
    out = {}
    for n, c in zip(names, contrib):
        raw = n.split("__", 1)[1]
        base = next((col for col in CATEGORICAL if raw.startswith(col + "_")), raw)
        out[base] = out.get(base, 0.0) + float(c)
    return out


def income_benchmark(income_model, emp):
    x = pd.DataFrame([{k: emp[k] for k in INCOME_PREDICTORS}])
    return float(income_model.predict(x)[0])


def recommend(emp, models, p=None):
    """emp: dict of raw employee fields. models: dict loaded from model.joblib.
    Returns probability, band, segment, pay gap and a ranked action list."""
    row = pd.DataFrame([emp])[CATEGORICAL + NUMERIC]
    clf = models["classifier"]
    if p is not None:
        p = float(p)  # pre-computed in a batch (workforce scoring)
    elif "ensemble" in models:
        # Risk score from the ensemble; the logistic model still explains the drivers
        ens_row = add_features(pd.DataFrame([emp]))[CATEGORICAL + NUMERIC + ENGINEERED]
        p = float(models["ensemble"].predict_proba(ens_row)[0, 1])
    else:
        p = float(clf.predict_proba(row)[0, 1])

    bench = income_benchmark(models["income_model"], emp)
    gap = (emp["MonthlyIncome"] - bench) / bench * 100

    contrib = feature_contributions(clf, row, models["train_mean"])

    seg_x = pd.DataFrame([{**emp, "OverTimeFlag": int(emp["OverTime"] == "Yes")}])[CLUSTER_FEATURES]
    seg = int(models["kmeans"].predict(models["cluster_scaler"].transform(seg_x))[0])
    seg_name = models["segment_names"][seg]

    ctx = {"pay_gap_pct": gap}
    actions = []
    for name, feats, test, action, owner, weight in RULES:
        if test(emp, ctx):
            impact = sum(max(contrib.get(f, 0.0), 0.0) for f in feats)
            actions.append({"driver": name, "action": action, "owner": owner,
                            "impact": round(impact, 3), "weight": weight,
                            "score": round(weight + 2 * impact, 3)})
    actions.sort(key=lambda a: a["score"], reverse=True)

    band = risk_band(p)
    if band == "Low":
        actions = actions[:2]
    elif band == "Medium":
        actions = actions[:4]
    else:
        actions = actions[:6]

    top_drivers = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {"probability": p, "band": band, "segment": seg,
            "segment_name": seg_name, "income_benchmark": bench,
            "pay_gap_pct": gap, "actions": actions,
            "top_drivers": [(k, round(v, 3)) for k, v in top_drivers if v > 0]}
