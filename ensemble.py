"""Ensemble model experiments for the Employee Retention Recommender.

Compares 18 combinations of single models, class balancing (SMOTE / class weights), feature
engineering, soft voting and stacking with repeated stratified 5-fold
cross-validation on the training set, then scores every model ONCE on the
untouched 25% hold-out test set (the same split as analysis.py).

Run:  python ensemble.py      (about 5-10 minutes)
Outputs: ensemble_results.json, figures/17_*.png, figures/18_*.png,
         and adds the chosen ensemble to model.joblib
"""
import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from recommender import CATEGORICAL, ENGINEERED, NUMERIC, add_features, load_clean

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SEED = 42

df = add_features(load_clean(os.path.join(HERE, "data", "hr_attrition.csv")))
y = df["AttritionFlag"]
X = df[CATEGORICAL + NUMERIC + ENGINEERED]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()  # class-imbalance ratio, about 5.2


def prep(engineered=True):
    num = NUMERIC + (ENGINEERED if engineered else [])
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL)])


def pipe(est, engineered=True, smote=False):
    if smote:
        return ImbPipeline([("prep", prep(engineered)), ("smote", SMOTE(random_state=SEED)), ("clf", est)])
    return Pipeline([("prep", prep(engineered)), ("clf", est)])


lr = lambda **k: LogisticRegression(max_iter=3000, C=0.3, **k)
svm = lambda: SVC(C=0.5, kernel="rbf", gamma="scale", probability=True, random_state=SEED)
rf = lambda: RandomForestClassifier(n_estimators=600, min_samples_leaf=3, max_features="sqrt",
                                    class_weight="balanced_subsample", n_jobs=1, random_state=SEED)
et = lambda: ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3, class_weight="balanced",
                                  n_jobs=1, random_state=SEED)
gb = lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=2,
                                        subsample=0.8, random_state=SEED)
xgb = lambda: XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=2, subsample=0.8,
                            colsample_bytree=0.6, min_child_weight=3, reg_lambda=2,
                            scale_pos_weight=spw, eval_metric="auc", n_jobs=1, random_state=SEED)
lgbm = lambda: LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2,
                              class_weight="balanced", verbose=-1, n_jobs=1, random_state=SEED)
cat = lambda: CatBoostClassifier(iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=5,
                                 auto_class_weights="Balanced", verbose=0, thread_count=1, random_seed=SEED)

EXPERIMENTS = [
    # name, family, pipeline
    ("Logistic regression (baseline)", "Single", pipe(lr(), engineered=False)),
    ("Logistic regression + engineered features", "Single", pipe(lr())),
    ("Logistic regression + class weights", "Balancing", pipe(lr(class_weight="balanced"))),
    ("Logistic regression + SMOTE", "Balancing", pipe(lr(), smote=True)),
    ("SVM (RBF kernel)", "Single", pipe(svm())),
    ("Random forest", "Bagging", pipe(rf())),
    ("Extra trees", "Bagging", pipe(et())),
    ("Gradient boosting", "Boosting", pipe(gb())),
    ("XGBoost", "Boosting", pipe(xgb())),
    ("XGBoost + SMOTE", "Boosting", pipe(XGBClassifier(**{**xgb().get_params(), "scale_pos_weight": 1}), smote=True)),
    ("LightGBM", "Boosting", pipe(lgbm())),
    ("CatBoost", "Boosting", pipe(cat())),
    ("Soft voting: LR + SVM + CatBoost", "Ensemble", pipe(VotingClassifier(
        [("lr", lr()), ("svm", svm()), ("cat", cat())], voting="soft"))),
    ("Soft voting: LR + XGBoost + LightGBM + RF", "Ensemble", pipe(VotingClassifier(
        [("lr", lr()), ("xgb", xgb()), ("lgbm", lgbm()), ("rf", rf())], voting="soft"))),
    ("Weighted voting: 2 LR + SVM + CatBoost", "Ensemble", pipe(VotingClassifier(
        [("lr", lr()), ("svm", svm()), ("cat", cat())], voting="soft", weights=[2, 1, 1]))),
    ("Soft voting: LR + LDA + SVM + Gradient boosting", "Ensemble", pipe(VotingClassifier(
        [("lr", lr()), ("lda", LinearDiscriminantAnalysis()), ("svm", svm()), ("gb", gb())], voting="soft"))),
    ("Stacking: LR + LDA + SVM + GB + XGBoost -> LR", "Ensemble", pipe(StackingClassifier(
        [("lr", lr()), ("lda", LinearDiscriminantAnalysis()), ("svm", svm()), ("gb", gb()), ("xgb", xgb())],
        final_estimator=LogisticRegression(max_iter=2000), stack_method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1))),
    ("Stacking: LR + SVM + RF + XGBoost + CatBoost -> LR", "Ensemble", pipe(StackingClassifier(
        [("lr", lr()), ("svm", svm()), ("rf", rf()), ("xgb", xgb()), ("cat", cat())],
        final_estimator=LogisticRegression(max_iter=2000), stack_method="predict_proba",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1))),
]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
rows, fitted = [], {}
for name, family, model in EXPERIMENTS:
    t0 = time.time()
    cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=4)
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    fitted[name] = (model, prob)
    rows.append({"Model": name, "Family": family,
                 "CV_AUC": round(cv_auc.mean(), 3), "CV_SD": round(cv_auc.std(), 3),
                 "Test_AUC": round(roc_auc_score(y_te, prob), 3),
                 "Train_AUC": round(roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]), 3),
                 "Seconds": round(time.time() - t0, 1)})
    print(f"{name:55s} CV {rows[-1]['CV_AUC']:.3f}±{rows[-1]['CV_SD']:.3f}  "
          f"test {rows[-1]['Test_AUC']:.3f}  train {rows[-1]['Train_AUC']:.3f}  ({rows[-1]['Seconds']}s)", flush=True)

res = pd.DataFrame(rows)
# Choose on cross-validated AUC (never on the test set), among ensembles
ens = res[res["Family"] == "Ensemble"].sort_values("CV_AUC", ascending=False)
best_name = ens.iloc[0]["Model"]
best_model, best_prob = fitted[best_name]
base_prob = fitted["Logistic regression (baseline)"][1]

# Threshold chosen by out-of-fold predictions on the training set (Youden's J)
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(best_model, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        method="predict_proba", n_jobs=4)[:, 1]
fpr, tpr, thr = roc_curve(y_tr, oof)
threshold = float(thr[np.argmax(tpr - fpr)])
pred = (best_prob >= threshold).astype(int)
best_metrics = {"Model": best_name, "Threshold": round(threshold, 3),
                "Test_AUC": round(roc_auc_score(y_te, best_prob), 3),
                "Accuracy": round(accuracy_score(y_te, pred), 3),
                "Precision": round(precision_score(y_te, pred), 3),
                "Recall": round(recall_score(y_te, pred), 3),
                "F1": round(f1_score(y_te, pred), 3)}

# Bootstrap 95% interval for the test AUC: how much the score moves by chance
rng = np.random.RandomState(SEED)
yt = y_te.values
boots = []
for _ in range(1000):
    i = rng.randint(0, len(yt), len(yt))
    if yt[i].min() != yt[i].max():
        boots.append(roc_auc_score(yt[i], best_prob[i]))
best_metrics["Test_AUC_CI"] = [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]

# Leakage demo: scoring on the training data itself inflates the AUC
rf_train_auc = float(res.loc[res["Model"] == "Random forest", "Train_AUC"].iloc[0])

out = {"experiments": rows, "best": best_metrics, "rf_train_auc": rf_train_auc,
       "baseline_test_auc": round(roc_auc_score(y_te, base_prob), 3),
       "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "test_leavers": int(y_te.sum())}
with open(os.path.join(HERE, "ensemble_results.json"), "w") as fh:
    json.dump(out, fh, indent=2)

# ---- Figures
plot = res.sort_values("CV_AUC")
colors = {"Single": "#94a3b8", "Balancing": "#94a3b8", "Bagging": "#1baf7a",
          "Boosting": "#2a78d6", "Ensemble": "#eb6834"}
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(plot["Model"], plot["CV_AUC"], xerr=plot["CV_SD"], color=[colors[f] for f in plot["Family"]],
        height=0.6, error_kw=dict(ecolor="#475569", capsize=3))
ax.scatter(plot["Test_AUC"], plot["Model"], color="#0f172a", zorder=3, s=22, label="Hold-out test AUC")
ax.set_xlim(0.6, 0.9); ax.set_xlabel("AUC (bar = 5-fold CV mean ± SD on training set; dot = test set)")
ax.set_title(f"{len(res)} model combinations compared")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
ax.legend(handles + [ax.collections[-1]], list(colors) + ["Test AUC"], loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "17_ensemble_comparison.png"), dpi=150); plt.close()

plt.figure(figsize=(7, 6))
for name, color in [(best_name, "#eb6834"), ("Logistic regression (baseline)", "#2a78d6"),
                    ("Random forest", "#1baf7a")]:
    p = fitted[name][1]
    f, t, _ = roc_curve(y_te, p)
    plt.plot(f, t, color=color, lw=2, label=f"{name.split(':')[0]} (AUC {roc_auc_score(y_te, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC on the hold-out test set")
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(FIG, "18_ensemble_roc.png"), dpi=150); plt.close()

# ---- Save the chosen ensemble next to the existing models
bundle_path = os.path.join(HERE, "model.joblib")
bundle = joblib.load(bundle_path)
bundle["ensemble"] = best_model
bundle["ensemble_name"] = best_name
bundle["ensemble_threshold"] = threshold
joblib.dump(bundle, bundle_path)
print(json.dumps(best_metrics, indent=1))
