"""Employee Retention Recommender: full statistical analysis and model build.

Run:  python analysis.py
Outputs: figures/*.png, results.json, model.joblib
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from statsmodels.stats.outliers_influence import variance_inflation_factor

from recommender import (CATEGORICAL, CLUSTER_FEATURES, INCOME_PREDICTORS,
                         NUMERIC, load_clean, recommend)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42
sns.set_theme(style="whitegrid", palette="deep")
R = {}


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=150)
    plt.close()


# ---------------------------------------------------------------- 1. Data
df = load_clean(os.path.join(HERE, "data", "hr_attrition.csv"))
R["n_rows"], R["n_cols_raw"] = len(df), 35
R["missing"] = int(df.isna().sum().sum())
R["attrition_rate"] = round(df["AttritionFlag"].mean() * 100, 2)
R["attrition_counts"] = df["Attrition"].value_counts().to_dict()

# ------------------------------------------------------ 2. Descriptive stats
desc_cols = ["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears",
             "YearsAtCompany", "YearsSinceLastPromotion", "JobSatisfaction",
             "WorkLifeBalance"]
desc = df[desc_cols].describe().T[["mean", "50%", "std", "min", "max"]]
desc["skew"] = df[desc_cols].skew()
R["descriptive"] = desc.round(2).reset_index().rename(
    columns={"index": "Variable", "50%": "median"}).to_dict("records")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df["Attrition"].value_counts().plot.bar(ax=ax[0], color=["#4C72B0", "#DD5555"])
ax[0].set_title("Attrition count"); ax[0].set_xlabel(""); ax[0].tick_params(rotation=0)
df["Attrition"].value_counts().plot.pie(ax=ax[1], autopct="%1.1f%%",
                                         colors=["#4C72B0", "#DD5555"])
ax[1].set_ylabel(""); ax[1].set_title("Attrition share")
save("01_attrition_distribution.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(data=df, x="Age", hue="Attrition", kde=True, ax=ax[0], bins=25)
ax[0].set_title("Age distribution by attrition")
sns.histplot(data=df, x="MonthlyIncome", hue="Attrition", kde=True, ax=ax[1], bins=25)
ax[1].set_title("Monthly income distribution by attrition")
save("02_histograms.png")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, col in zip(ax, ["MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"]):
    sns.boxplot(data=df, x="Attrition", y=col, ax=a)
    a.set_title(f"{col} by attrition")
save("03_boxplots.png")

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for a, col in zip(ax.ravel(), ["OverTime", "MaritalStatus", "BusinessTravel", "JobRole"]):
    rate = df.groupby(col)["AttritionFlag"].mean().sort_values() * 100
    rate.plot.barh(ax=a, color="#DD5555")
    a.set_title(f"Attrition rate (%) by {col}"); a.set_ylabel("")
save("04_attrition_rate_by_category.png")

sns.scatterplot(data=df, x="TotalWorkingYears", y="MonthlyIncome",
                hue="Attrition", alpha=0.6)
plt.title("Experience vs monthly income")
save("05_scatter_experience_income.png")

corr_cols = NUMERIC + ["AttritionFlag", "OverTimeFlag"]
corr = df[corr_cols].corr()
plt.figure(figsize=(14, 11))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, square=True)
plt.title("Correlation matrix")
save("06_correlation_heatmap.png")
R["top_corr_with_attrition"] = corr["AttritionFlag"].drop("AttritionFlag").sort_values(
    key=abs, ascending=False).head(8).round(3).to_dict()

# Treemap-style view: headcount and attrition by department / role
grp = df.groupby(["Department", "JobRole"]).agg(
    Headcount=("AttritionFlag", "size"), Leavers=("AttritionFlag", "sum")).reset_index()
grp["Rate"] = (grp["Leavers"] / grp["Headcount"] * 100).round(1)
R["role_table"] = grp.sort_values("Rate", ascending=False).to_dict("records")

# --------------------------------------------------- 3. Hypothesis testing
chi = []
for col in CATEGORICAL + ["JobLevel", "StockOptionLevel", "JobSatisfaction",
                          "EnvironmentSatisfaction", "WorkLifeBalance", "JobInvolvement"]:
    tab = pd.crosstab(df[col], df["Attrition"])
    c2, p, dof, _ = stats.chi2_contingency(tab)
    n = tab.values.sum()
    v = np.sqrt(c2 / (n * (min(tab.shape) - 1)))
    chi.append({"Variable": col, "Chi2": round(c2, 2), "df": dof,
                "p_value": float(f"{p:.3g}"), "Cramers_V": round(v, 3),
                "Significant": "Yes" if p < 0.05 else "No"})
R["chi_square"] = sorted(chi, key=lambda r: r["p_value"])

tt = []
for col in ["Age", "MonthlyIncome", "DistanceFromHome", "TotalWorkingYears",
            "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager",
            "YearsSinceLastPromotion", "NumCompaniesWorked",
            "PercentSalaryHike", "TrainingTimesLastYear", "DailyRate", "HourlyRate"]:
    a = df.loc[df.Attrition == "Yes", col]
    b = df.loc[df.Attrition == "No", col]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b)
    tt.append({"Variable": col, "Mean_Leavers": round(a.mean(), 2),
               "Mean_Stayers": round(b.mean(), 2), "t": round(t, 2),
               "p_value_t": float(f"{p:.3g}"), "p_value_MW": float(f"{pu:.3g}"),
               "Significant": "Yes" if p < 0.05 else "No"})
R["t_tests"] = sorted(tt, key=lambda r: r["p_value_t"])

# One-way ANOVA: income across job levels (supports the salary benchmark)
groups = [g["MonthlyIncome"].values for _, g in df.groupby("JobLevel")]
f, p = stats.f_oneway(*groups)
R["anova_income_joblevel"] = {"F": round(f, 2), "p_value": float(f"{p:.3g}")}

# Normality check on income (Shapiro-Wilk on a sample)
w, p = stats.shapiro(df["MonthlyIncome"].sample(500, random_state=SEED))
R["shapiro_income"] = {"W": round(w, 3), "p_value": float(f"{p:.3g}")}

# ---------------------------------------------- 4. Regression (salary benchmark)
X1 = sm.add_constant(df[["JobLevel"]])
m1 = sm.OLS(df["MonthlyIncome"], X1).fit()
Xm = sm.add_constant(df[INCOME_PREDICTORS])
m2 = sm.OLS(df["MonthlyIncome"], Xm).fit()
R["slr"] = {"r2": round(m1.rsquared, 3), "adj_r2": round(m1.rsquared_adj, 3),
            "aic": round(m1.aic, 1), "coef": m1.params.round(2).to_dict()}
R["mlr"] = {"r2": round(m2.rsquared, 3), "adj_r2": round(m2.rsquared_adj, 3),
            "aic": round(m2.aic, 1), "f": round(m2.fvalue, 1),
            "f_p": float(f"{m2.f_pvalue:.3g}"),
            "coef": [{"Term": k, "Coef": round(m2.params[k], 2),
                      "Std_Err": round(m2.bse[k], 2), "t": round(m2.tvalues[k], 2),
                      "p_value": float(f"{m2.pvalues[k]:.3g}")} for k in m2.params.index]}
vif = [{"Variable": c, "VIF": round(variance_inflation_factor(Xm.values, i), 2)}
       for i, c in enumerate(Xm.columns) if c != "const"]
R["vif"] = vif
dw = sm.stats.stattools.durbin_watson(m2.resid)
R["mlr"]["durbin_watson"] = round(dw, 3)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(m2.fittedvalues, m2.resid, alpha=0.4)
ax[0].axhline(0, color="red"); ax[0].set_xlabel("Fitted income"); ax[0].set_ylabel("Residual")
ax[0].set_title("Residuals vs fitted (MLR)")
sm.qqplot(m2.resid, line="s", ax=ax[1]); ax[1].set_title("Normal Q-Q of residuals")
save("07_regression_diagnostics.png")

income_model = LinearRegression().fit(df[INCOME_PREDICTORS], df["MonthlyIncome"])
df["IncomeBenchmark"] = income_model.predict(df[INCOME_PREDICTORS])
df["PayGapPct"] = (df["MonthlyIncome"] - df["IncomeBenchmark"]) / df["IncomeBenchmark"] * 100
under = df["PayGapPct"] <= -15
R["underpaid_share"] = round(under.mean() * 100, 1)
R["attr_underpaid"] = round(df.loc[under, "AttritionFlag"].mean() * 100, 1)
R["attr_not_underpaid"] = round(df.loc[~under, "AttritionFlag"].mean() * 100, 1)
c2, p, _, _ = stats.chi2_contingency(pd.crosstab(under, df["AttritionFlag"]))
R["underpaid_chi_p"] = float(f"{p:.3g}")

# ------------------------------------------------------ 5. Classification
X = df[CATEGORICAL + NUMERIC]
y = df["AttritionFlag"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=SEED)
R["train_n"], R["test_n"] = len(X_tr), len(X_te)


def prep():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                              sparse_output=False), CATEGORICAL)])


knn_grid = GridSearchCV(Pipeline([("prep", prep()), ("clf", KNeighborsClassifier())]),
                        {"clf__n_neighbors": list(range(3, 31, 2))},
                        scoring="roc_auc", cv=5).fit(X_tr, y_tr)
best_k = knn_grid.best_params_["clf__n_neighbors"]
tree_grid = GridSearchCV(Pipeline([("prep", prep()),
                                   ("clf", DecisionTreeClassifier(random_state=SEED))]),
                         {"clf__max_depth": [3, 4, 5, 6, 8],
                          "clf__min_samples_leaf": [5, 10, 20]},
                         scoring="roc_auc", cv=5).fit(X_tr, y_tr)
R["knn_k"] = best_k
R["tree_params"] = {k.replace("clf__", ""): v for k, v in tree_grid.best_params_.items()}

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED, **R["tree_params"]),
    "KNN": KNeighborsClassifier(n_neighbors=best_k),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
}
fitted, perf = {}, []
fig_roc = plt.figure(figsize=(7, 6))
fig_cm, axes_cm = plt.subplots(1, 5, figsize=(20, 4))
for (name, est), axc in zip(models.items(), axes_cm):
    pipe = Pipeline([("prep", prep()), ("clf", est)]).fit(X_tr, y_tr)
    fitted[name] = pipe
    prob = pipe.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred)
    perf.append({"Model": name, "Accuracy": round(accuracy_score(y_te, pred), 3),
                 "Precision": round(precision_score(y_te, pred, zero_division=0), 3),
                 "Recall": round(recall_score(y_te, pred), 3),
                 "F1": round(f1_score(y_te, pred), 3),
                 "AUC": round(roc_auc_score(y_te, prob), 3),
                 "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
                 "FN": int(cm[1, 0]), "TP": int(cm[1, 1])})
    fpr, tpr, _ = roc_curve(y_te, prob)
    plt.figure(fig_roc.number)
    plt.plot(fpr, tpr, label=f"{name} (AUC={perf[-1]['AUC']:.3f})")
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=axc,
                xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
    axc.set_title(name); axc.set_xlabel("Predicted"); axc.set_ylabel("Actual")
plt.figure(fig_roc.number)
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate")
plt.ylabel("True positive rate"); plt.title("ROC curves (test set)"); plt.legend()
save("08_roc_curves.png")
plt.figure(fig_cm.number); save("09_confusion_matrices.png")
R["classifier_performance"] = perf
best = max(perf, key=lambda r: r["AUC"])["Model"]
R["best_model"] = best

# Logistic regression with a tuned threshold (Youden's J) for the deployed model
lr_pipe = fitted["Logistic Regression"]
prob_lr = lr_pipe.predict_proba(X_te)[:, 1]
fpr, tpr, thr = roc_curve(y_te, prob_lr)
j = int(np.argmax(tpr - fpr))
R["lr_youden_threshold"] = round(float(thr[j]), 3)
pred_j = (prob_lr >= thr[j]).astype(int)
R["lr_tuned"] = {"Accuracy": round(accuracy_score(y_te, pred_j), 3),
                 "Precision": round(precision_score(y_te, pred_j), 3),
                 "Recall": round(recall_score(y_te, pred_j), 3),
                 "F1": round(f1_score(y_te, pred_j), 3)}

# Odds ratios from statsmodels logit on key drivers (interpretation)
key = ["OverTimeFlag", "JobSatisfaction", "EnvironmentSatisfaction",
       "WorkLifeBalance", "JobInvolvement", "StockOptionLevel",
       "YearsSinceLastPromotion", "DistanceFromHome", "NumCompaniesWorked",
       "TotalWorkingYears", "YearsAtCompany", "Age"]
lg = df[key].copy()
lg["MonthlyIncome_k"] = df["MonthlyIncome"] / 1000
lg["Single"] = (df["MaritalStatus"] == "Single").astype(int)
lg["TravelFrequently"] = (df["BusinessTravel"] == "Travel_Frequently").astype(int)
logit = sm.Logit(y, sm.add_constant(lg)).fit(disp=0)
R["logit"] = {"pseudo_r2": round(logit.prsquared, 3),
              "llr_p": float(f"{logit.llr_pvalue:.3g}"),
              "terms": [{"Term": k, "Coef": round(logit.params[k], 3),
                         "Odds_Ratio": round(float(np.exp(logit.params[k])), 3),
                         "p_value": float(f"{logit.pvalues[k]:.3g}")}
                        for k in logit.params.index if k != "const"]}

coef_names = lr_pipe.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lr_pipe.named_steps["clf"].coef_[0],
                  index=[n.split("__", 1)[1] for n in coef_names])
top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15)
plt.figure(figsize=(8, 6))
top.sort_values().plot.barh(color=["#DD5555" if v > 0 else "#4C72B0" for v in top.sort_values()])
plt.title("Top logistic regression coefficients (standardised)")
plt.xlabel("Coefficient (red raises attrition risk)")
save("10_lr_coefficients.png")

plt.figure(figsize=(18, 8))
plot_tree(fitted["Decision Tree"].named_steps["clf"], max_depth=3, filled=True,
          feature_names=[n.split("__", 1)[1] for n in coef_names],
          class_names=["Stay", "Leave"], fontsize=8, impurity=False, proportion=True)
plt.title("Decision tree (top 3 levels)")
save("11_decision_tree.png")

# ---------------------------------------- 6. Dimensionality reduction (PCA, FA)
sc = StandardScaler().fit(df[NUMERIC])
Z = sc.transform(df[NUMERIC])
pca = PCA().fit(Z)
ev = pca.explained_variance_ratio_
R["pca_var"] = [round(v * 100, 2) for v in ev[:10]]
R["pca_n_80"] = int(np.argmax(np.cumsum(ev) >= 0.80) + 1)
R["pca_kaiser"] = int((pca.explained_variance_ > 1).sum())
loadings = pd.DataFrame(pca.components_[:3].T, index=NUMERIC, columns=["PC1", "PC2", "PC3"])
R["pca_top_loadings"] = {pc: loadings[pc].abs().sort_values(ascending=False).head(5).index.tolist()
                         for pc in loadings}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(range(1, 16), ev[:15] * 100); ax[0].plot(range(1, 16), np.cumsum(ev[:15]) * 100, "ro-")
ax[0].set_xlabel("Component"); ax[0].set_ylabel("% variance"); ax[0].set_title("PCA scree plot")
pcs = pca.transform(Z)[:, :2]
ax[1].scatter(pcs[:, 0], pcs[:, 1], c=df["AttritionFlag"], cmap="coolwarm", alpha=0.5, s=10)
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].set_title("Employees on PC1-PC2 (red = left)")
save("12_pca.png")

fa = FactorAnalysis(n_components=3, rotation="varimax", random_state=SEED).fit(Z)
fal = pd.DataFrame(fa.components_.T, index=NUMERIC, columns=["F1", "F2", "F3"])
R["fa_top_loadings"] = {f: fal[f].abs().sort_values(ascending=False).head(4).index.tolist()
                        for f in fal}

# ----------------------------------------------- 7. Clustering (segments)
csc = StandardScaler().fit(df[CLUSTER_FEATURES])
C = csc.transform(df[CLUSTER_FEATURES])
inertia, sil = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(C)
    inertia.append(km.inertia_); sil.append(silhouette_score(C, km.labels_))
R["silhouette"] = {k: round(s, 3) for k, s in zip(range(2, 9), sil)}
K = 4
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(2, 9), inertia, "o-"); ax[0].axvline(K, ls="--", c="grey")
ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(range(2, 9), sil, "o-"); ax[1].axvline(K, ls="--", c="grey")
ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
save("13_kmeans_selection.png")

kmeans = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(C)
df["Segment"] = kmeans.labels_
prof = df.groupby("Segment").agg(
    Employees=("AttritionFlag", "size"), AttritionRate=("AttritionFlag", "mean"),
    Age=("Age", "mean"), MonthlyIncome=("MonthlyIncome", "mean"),
    TotalWorkingYears=("TotalWorkingYears", "mean"),
    YearsAtCompany=("YearsAtCompany", "mean"),
    YearsSinceLastPromotion=("YearsSinceLastPromotion", "mean"),
    JobSatisfaction=("JobSatisfaction", "mean"),
    EnvironmentSatisfaction=("EnvironmentSatisfaction", "mean"),
    WorkLifeBalance=("WorkLifeBalance", "mean"),
    OverTimeShare=("OverTimeFlag", "mean"))
prof["AttritionRate"] *= 100; prof["OverTimeShare"] *= 100


def name_segment(r):
    if r["YearsAtCompany"] > 12:
        return "Senior long-tenure staff"
    if r["TotalWorkingYears"] > 18:
        return "Experienced lateral hires"
    if r["OverTimeShare"] > 60:
        return "Overtime-heavy staff"
    if r["JobSatisfaction"] < 2.4 or r["EnvironmentSatisfaction"] < 2.4:
        return "Disengaged staff"
    if r["Age"] < 33:
        return "Young early-career staff"
    return "Core staff (no overtime)"


names, used = {}, set()
for s, r in prof.iterrows():
    n = name_segment(r)
    while n in used:
        n += " (B)"
    used.add(n); names[int(s)] = n
prof["Name"] = prof.index.map(names)
R["segments"] = prof.round(2).reset_index().to_dict("records")
c2, p, _, _ = stats.chi2_contingency(pd.crosstab(df["Segment"], df["AttritionFlag"]))
R["segment_chi_p"] = float(f"{p:.3g}")

plt.figure(figsize=(7, 4))
(prof.set_index("Name")["AttritionRate"].sort_values()).plot.barh(color="#DD5555")
plt.xlabel("Attrition rate (%)"); plt.title("Attrition rate by K-means segment"); plt.ylabel("")
save("14_segment_attrition.png")

sample = C[np.random.RandomState(SEED).choice(len(C), 200, replace=False)]
plt.figure(figsize=(12, 5))
dendrogram(linkage(sample, "ward"), no_labels=True, color_threshold=None)
plt.title("Hierarchical clustering dendrogram (Ward, 200-employee sample)")
save("15_dendrogram.png")

# ------------------------------------------- 8. Recommendations on the dataset
pre = lr_pipe.named_steps["prep"]
train_mean = np.asarray(pre.transform(X_tr)).mean(axis=0)
bundle = {"classifier": lr_pipe, "income_model": income_model,
          "train_mean": train_mean, "kmeans": kmeans, "cluster_scaler": csc,
          "segment_names": names, "threshold": R["lr_youden_threshold"],
          "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL}}
joblib.dump(bundle, os.path.join(HERE, "model.joblib"))

current = df[df["Attrition"] == "No"].copy()
recs = [recommend(r[CATEGORICAL + NUMERIC].to_dict(), bundle) for _, r in current.iterrows()]
current["Risk"] = [r["probability"] for r in recs]
current["Band"] = [r["band"] for r in recs]
R["band_counts"] = current["Band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int).to_dict()
from collections import Counter
cnt = Counter(a["driver"] for r, b in zip(recs, current["Band"]) if b != "Low" for a in r["actions"])
R["action_frequency"] = [{"Driver": k, "Employees": v} for k, v in cnt.most_common(10)]
R["dept_risk"] = (current.groupby("Department")["Risk"].mean() * 100).round(1).to_dict()

top_risk = current.sort_values("Risk", ascending=False).head(5)
R["sample_recommendations"] = []
for idx, r in top_risk.iterrows():
    rec = recommend(r[CATEGORICAL + NUMERIC].to_dict(), bundle)
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
save("16_action_frequency.png")

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(R, fh, indent=2, default=str)
print(json.dumps({k: R[k] for k in ["attrition_rate", "best_model", "classifier_performance",
                                    "lr_youden_threshold", "lr_tuned", "mlr", "band_counts",
                                    "segments", "silhouette", "underpaid_share",
                                    "attr_underpaid", "attr_not_underpaid", "action_frequency"]},
                 indent=1, default=str))
