"""Employee Retention Recommender: Streamlit web app.

Run locally:  streamlit run app.py
"""
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from recommender import (CATEGORICAL, CLUSTER_FEATURES, ENGINEERED, NUMERIC, add_features,
                         load_clean, recommend)

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Employee Retention Recommender", page_icon="◆", layout="wide")

# ---------------------------------------------------------------- Design tokens
INK, INK_2, MUTED = "#0f172a", "#475569", "#94a3b8"
SURFACE, LINE, ACCENT = "#ffffff", "#e2e8f0", "#2a78d6"
STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
# Categorical slots 1-3 for the three smaller segments; the large baseline
# segment stays neutral grey so a 3D scatter never needs a fourth hue.
SEGMENT_COLOURS = {"Overtime-heavy staff": "#eb6834", "Senior long-tenure staff": "#2a78d6",
                   "Experienced lateral hires": "#1baf7a", "Core staff (no overtime)": "#b6bcc6"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: 'Inter', system-ui, sans-serif !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.02em; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}

.hero {{ position: relative; padding: 34px 36px; border-radius: 20px; margin-bottom: 18px;
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #2a78d6 140%); color: #fff;
  box-shadow: 0 1px 2px rgba(15,23,42,.08), 0 24px 48px -24px rgba(15,23,42,.45);
  overflow: hidden; }}
.hero:after {{ content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
  border-radius: 40px; transform: rotate(28deg); background: rgba(255,255,255,.06);
  box-shadow: -40px 70px 0 rgba(255,255,255,.04); }}
.eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
  color: #9cc3f2; }}
.hero h1 {{ color: #fff; font-size: 34px; font-weight: 700; margin: 6px 0 6px; padding: 0; }}
.hero p {{ color: #cbd5e1; font-size: 15px; max-width: 720px; margin: 0; }}
.chips {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }}
.chip {{ background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
  border-radius: 12px; padding: 8px 14px; font-size: 13px; color: #e2e8f0; }}
.chip b {{ color: #fff; font-size: 15px; margin-right: 4px; }}

.scene {{ perspective: 900px; }}
.card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.18);
  transition: transform .25s ease, box-shadow .25s ease; transform-style: preserve-3d; min-height: 158px; }}
.card:hover {{ transform: rotateX(4deg) rotateY(-4deg) translateY(-3px);
  box-shadow: 0 2px 4px rgba(15,23,42,.06), 0 22px 40px -16px rgba(15,23,42,.30); }}
.label {{ font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: {INK_2}; }}
.value {{ font-size: 30px; font-weight: 700; color: {INK}; margin-top: 6px; line-height: 1.15; }}
.value.sm {{ font-size: 19px; }}
.sub {{ font-size: 13px; color: {INK_2}; margin-top: 4px; }}
.pill {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600;
  padding: 3px 10px; border-radius: 999px; background: #f1f5f9; color: {INK}; margin-top: 8px; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.meter {{ height: 8px; background: #eef2f7; border-radius: 999px; margin-top: 12px; overflow: hidden; }}
.meter span {{ display: block; height: 100%; border-radius: 999px; }}

.action {{ display: flex; gap: 14px; padding: 14px 16px; border: 1px solid {LINE}; border-radius: 14px;
  background: {SURFACE}; margin-bottom: 10px; transition: transform .2s ease, box-shadow .2s ease; }}
.action:hover {{ transform: translateY(-2px); box-shadow: 0 12px 24px -16px rgba(15,23,42,.35); }}
.num {{ flex: none; width: 28px; height: 28px; border-radius: 8px; background: #eaf2fc; color: {ACCENT};
  font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center; }}
.atext {{ font-weight: 600; color: {INK}; font-size: 14.5px; }}
.ameta {{ color: {INK_2}; font-size: 12.5px; margin-top: 3px; }}
.section {{ font-size: 18px; font-weight: 600; color: {INK}; margin: 26px 0 10px; }}
.hint {{ color: {INK_2}; font-size: 13px; margin: -4px 0 10px; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 16px; font-weight: 500; color: {INK_2}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
[data-testid="stForm"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 16px;
  padding: 22px 24px; box-shadow: 0 8px 24px -16px rgba(15,23,42,.18); }}
.stButton button, [data-testid="stFormSubmitButton"] button {{ border-radius: 10px; font-weight: 600;
  padding: 10px 22px; }}
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; }}

/* ---------- Motion ---------- */
@keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes slideIn {{ from {{ opacity: 0; transform: translateX(-16px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes grow {{ from {{ width: 0; }} }}
@keyframes drift {{ 0% {{ background-position: 0% 50%; }} 100% {{ background-position: 100% 50%; }} }}
@keyframes float {{ 0%, 100% {{ transform: rotate(28deg) translateY(0); }} 50% {{ transform: rotate(34deg) translateY(14px); }} }}
@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 var(--pc); }} 70% {{ box-shadow: 0 0 0 7px transparent; }} 100% {{ box-shadow: 0 0 0 0 transparent; }} }}
@property --n {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@keyframes count {{ from {{ --n: 0; }} }}

.hero {{ background-size: 220% 220%; animation: fadeUp .7s ease-out both, drift 14s ease-in-out infinite alternate; }}
.hero:after {{ animation: float 9s ease-in-out infinite; }}
.hero h1 {{ animation: fadeUp .7s .1s ease-out both; }}
.hero p {{ animation: fadeUp .7s .2s ease-out both; }}
.chip {{ animation: fadeUp .5s ease-out both; transition: background .2s, transform .2s; }}
.chip:hover {{ background: rgba(255,255,255,.16); transform: translateY(-2px); }}
.chip:nth-child(1) {{ animation-delay: .30s; }} .chip:nth-child(2) {{ animation-delay: .38s; }}
.chip:nth-child(3) {{ animation-delay: .46s; }} .chip:nth-child(4) {{ animation-delay: .54s; }}
.card {{ animation: fadeUp .6s ease-out both; }}
.action {{ animation: slideIn .5s ease-out both; }}
.section, .hint {{ animation: fadeIn .6s ease-out both; }}
.meter span {{ animation: grow 1.2s cubic-bezier(.2,.8,.2,1) .3s both; }}
.count {{ counter-reset: n var(--n); animation: count 1.2s cubic-bezier(.2,.8,.2,1) .2s both; }}
.count::before {{ content: counter(n); }}
.dot.live {{ animation: pulse 2s ease-out infinite; }}
[data-testid="stTabs"] [role="tabpanel"] {{ animation: fadeIn .45s ease-out both; }}
[data-testid="stForm"] {{ animation: fadeUp .6s .15s ease-out both; }}
.stButton button, [data-testid="stFormSubmitButton"] button {{ transition: transform .15s ease, box-shadow .15s ease; }}
[data-testid="stFormSubmitButton"] button:hover {{ transform: translateY(-1px); box-shadow: 0 10px 20px -10px rgba(42,120,214,.7); }}
[data-testid="stPlotlyChart"] {{ animation: fadeUp .7s .2s ease-out both; }}
@media (prefers-reduced-motion: reduce) {{ *, *:before, *:after {{ animation: none !important; transition: none !important; }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- Data & models
@st.cache_resource
def load_models():
    return joblib.load(os.path.join(HERE, "model.joblib"))


@st.cache_data
def load_data():
    return load_clean(os.path.join(HERE, "data", "hr_attrition.csv"))


@st.cache_data
def load_results():
    with open(os.path.join(HERE, "results.json")) as fh:
        res = json.load(fh)
    ens_path = os.path.join(HERE, "ensemble_results.json")
    if os.path.exists(ens_path):
        with open(ens_path) as fh:
            res["ensemble"] = json.load(fh)
    return res


@st.cache_data(show_spinner="Scoring the workforce…")
def score_workforce():
    df, m = load_data(), load_models()
    if "ensemble" in m:  # score everyone in one batch: much faster than row by row
        probs = m["ensemble"].predict_proba(add_features(df)[CATEGORICAL + NUMERIC + ENGINEERED])[:, 1]
    else:
        probs = [None] * len(df)
    recs = [recommend(r[CATEGORICAL + NUMERIC].to_dict(), m, p) for (_, r), p in zip(df.iterrows(), probs)]
    df = df.copy()
    df["Risk %"] = [round(r["probability"] * 100, 1) for r in recs]
    df["Band"] = [r["band"] for r in recs]
    df["Segment"] = [r["segment_name"] for r in recs]
    df["Top action"] = [r["actions"][0]["action"] if r["actions"] else "No action needed" for r in recs]
    return df


@st.cache_resource
def fit_pca():
    m = load_models()
    return PCA(n_components=3, random_state=42).fit(
        m["cluster_scaler"].transform(load_data()[CLUSTER_FEATURES]))


models, data, results = load_models(), load_data(), load_results()
cats = models["categories"]
AXES = ["Seniority", "Overtime & environment", "Work-life balance"]


def pretty(name):
    """OverTime -> Over time, JobSatisfaction -> Job satisfaction."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).capitalize()


def card(label, value, sub="", small=False, extra="", delay=0.0):
    return (f'<div class="scene"><div class="card" style="animation-delay:{delay:.2f}s"><div class="label">{label}</div>'
            f'<div class="value{" sm" if small else ""}">{value}</div>'
            f'{f"<div class=sub>{sub}</div>" if sub else ""}{extra}</div></div>')


def band_pill(band):
    live = " live" if band == "High" else ""
    return (f'<span class="pill"><span class="dot{live}" style="background:{STATUS[band]};--pc:{STATUS[band]}88"></span>'
            f'{band} risk</span>')


def style_fig(fig, height=300):
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=INK_2, size=12),
                      hoverlabel=dict(bgcolor="#fff", bordercolor=LINE, font_color=INK))
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", linecolor=LINE)
    return fig


def hbar(series, unit="%", height=260):
    s = series.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=ACCENT,
                           marker_cornerradius=4, width=0.5,
                           text=[f"{v:.1f}{unit}" for v in s.values], textposition="outside",
                           textfont=dict(color=INK_2),
                           hovertemplate="%{y}: %{x:.2f}" + unit + "<extra></extra>"))
    fig.update_xaxes(range=[0, s.max() * 1.2])
    return style_fig(fig, height)


def map_3d(emp=None, emp_result=None, height=560, rotate=False):
    wf = score_workforce()
    pca = fit_pca()
    xyz = pca.transform(models["cluster_scaler"].transform(wf[CLUSTER_FEATURES]))
    fig = go.Figure()
    order = ["Core staff (no overtime)", "Senior long-tenure staff", "Experienced lateral hires",
             "Overtime-heavy staff"]
    for name in order:
        mask = (wf["Segment"] == name).values
        sub = wf[mask]
        fig.add_trace(go.Scatter3d(
            x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2], mode="markers", name=name,
            marker=dict(size=3.2 if name.startswith("Core") else 3.8, color=SEGMENT_COLOURS[name],
                        opacity=0.5 if name.startswith("Core") else 0.85,
                        line=dict(width=0)),
            customdata=sub[["JobRole", "Risk %", "Attrition"]].values,
            hovertemplate=("<b>%{customdata[0]}</b><br>Risk %{customdata[1]}%<br>"
                           "Left company: %{customdata[2]}<extra>" + name + "</extra>")))
    if emp is not None:
        e = pd.DataFrame([{**emp, "OverTimeFlag": int(emp["OverTime"] == "Yes")}])[CLUSTER_FEATURES]
        p = pca.transform(models["cluster_scaler"].transform(e))[0]
        fig.add_trace(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name="This employee",
            marker=dict(size=13, color=INK, symbol="diamond", line=dict(color="#fff", width=2)),
            hovertemplate=f"<b>This employee</b><br>Risk {emp_result['probability'] * 100:.1f}%<extra></extra>"))
    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#e2e8f0", showspikes=False,
                zerolinecolor="#cbd5e1", tickfont=dict(size=10, color=MUTED))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK_2),
        legend=dict(orientation="h", y=1.0, x=0, font=dict(size=12), itemsizing="constant"),
        scene=dict(xaxis=dict(title=AXES[0], **axis), yaxis=dict(title=AXES[1], **axis),
                   zaxis=dict(title=AXES[2], **axis),
                   camera=dict(eye=dict(x=1.25, y=1.15, z=0.7))),
        hoverlabel=dict(bgcolor="#fff", bordercolor=LINE, font_color=INK))
    if rotate:
        # Camera orbit frames: the Rotate button spins the scene once around the vertical axis
        r0, z0 = 1.7, 0.7
        fig.frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(
            x=r0 * np.cos(t), y=r0 * np.sin(t), z=z0)))) for t in np.linspace(0.72, 0.72 + 2 * np.pi, 90)]
        fig.update_layout(updatemenus=[dict(
            type="buttons", direction="left", showactive=False, x=1, xanchor="right", y=1.0, yanchor="bottom",
            bgcolor="#fff", bordercolor=LINE, font=dict(color=INK, size=12),
            buttons=[dict(label="▶  Rotate", method="animate",
                          args=[None, dict(frame=dict(duration=45, redraw=True),
                                           transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                     dict(label="❚❚  Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    return fig


# ---------------------------------------------------------------- Hero
lr_auc = next(p["AUC"] for p in results["classifier_performance"] if p["Model"] == "Logistic Regression")
if "ensemble" in results:
    lr_auc = results["ensemble"]["best"]["Test_AUC"]
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">HR Analytics · Ensemble recommendation model · Version 3</div>
  <h1>Employee Retention Recommender</h1>
  <p>Estimate an employee's risk of leaving, see what is driving it, and get ranked retention
  actions, backed by hypothesis tests, an ensemble classifier, regression benchmarks and K-means segments.</p>
  <div class="chips">
    <div class="chip"><b>{results['n_rows']:,}</b>employees analysed</div>
    <div class="chip"><b>{results['attrition_rate']}%</b>historical attrition</div>
    <div class="chip"><b>{lr_auc:.3f}</b>test AUC</div>
    <div class="chip"><b>4</b>employee segments</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Assess an employee", "Workforce overview",
                                  "3D workforce map", "Model insights"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    med = data[NUMERIC].median()
    with st.form("emp"):
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown('<div class="label">Personal</div>', unsafe_allow_html=True)
            age = st.slider("Age", 18, 60, 30)
            gender = st.selectbox("Gender", cats["Gender"])
            marital = st.selectbox("Marital status", cats["MaritalStatus"], index=cats["MaritalStatus"].index("Single"))
            distance = st.slider("Distance from home (km)", 1, 29, 10)
            education = st.select_slider("Education", [1, 2, 3, 4, 5], 3,
                                         format_func=lambda v: ["Below college", "College", "Bachelor", "Master", "Doctor"][v - 1])
            edu_field = st.selectbox("Education field", cats["EducationField"])
        with c2:
            st.markdown('<div class="label">Job & pay</div>', unsafe_allow_html=True)
            dept = st.selectbox("Department", cats["Department"], index=cats["Department"].index("Sales"))
            role = st.selectbox("Job role", cats["JobRole"], index=cats["JobRole"].index("Sales Executive"))
            level = st.select_slider("Job level", [1, 2, 3, 4, 5], 2)
            income = st.number_input("Monthly income ($)", 1000, 20000, 4500, step=100)
            overtime = st.radio("Works overtime?", ["No", "Yes"], index=1, horizontal=True)
            travel = st.selectbox("Business travel", cats["BusinessTravel"], index=cats["BusinessTravel"].index("Travel_Rarely"))
            stock = st.select_slider("Stock option level", [0, 1, 2, 3], 0)
            hike = st.slider("Last salary hike (%)", 11, 25, 14)
        with c3:
            st.markdown('<div class="label">Experience & engagement (1 = low, 4 = high)</div>', unsafe_allow_html=True)
            total_yrs = st.slider("Total working years", 0, 40, 6)
            yrs_co = st.slider("Years at company", 0, 40, 3)
            yrs_role = st.slider("Years in current role", 0, 18, 2)
            yrs_promo = st.slider("Years since last promotion", 0, 15, 1)
            yrs_mgr = st.slider("Years with current manager", 0, 17, 2)
            n_comp = st.slider("Number of companies worked", 0, 9, 2)
            js = st.select_slider("Job satisfaction", [1, 2, 3, 4], 2)
            es = st.select_slider("Environment satisfaction", [1, 2, 3, 4], 2)
            wlb = st.select_slider("Work-life balance", [1, 2, 3, 4], 3)
            ji = st.select_slider("Job involvement", [1, 2, 3, 4], 3)
            rs = st.select_slider("Relationship satisfaction", [1, 2, 3, 4], 3)
            perf = st.select_slider("Performance rating", [3, 4], 3)
            training = st.slider("Trainings last year", 0, 6, 2)
        submitted = st.form_submit_button("Get recommendation", type="primary")

    if submitted:
        emp = {"Age": age, "Gender": gender, "MaritalStatus": marital,
               "DistanceFromHome": distance, "Education": education,
               "EducationField": edu_field, "Department": dept, "JobRole": role,
               "JobLevel": level, "MonthlyIncome": income, "OverTime": overtime,
               "BusinessTravel": travel, "StockOptionLevel": stock,
               "PercentSalaryHike": hike, "TotalWorkingYears": total_yrs,
               "YearsAtCompany": yrs_co, "YearsInCurrentRole": yrs_role,
               "YearsSinceLastPromotion": yrs_promo, "YearsWithCurrManager": yrs_mgr,
               "NumCompaniesWorked": n_comp, "JobSatisfaction": js,
               "EnvironmentSatisfaction": es, "WorkLifeBalance": wlb,
               "JobInvolvement": ji, "RelationshipSatisfaction": rs,
               "PerformanceRating": perf, "TrainingTimesLastYear": training,
               # Rates with no predictive value are held at the dataset median
               "DailyRate": int(med["DailyRate"]), "HourlyRate": int(med["HourlyRate"]),
               "MonthlyRate": int(med["MonthlyRate"])}
        st.session_state["employee"] = emp
        st.session_state["result"] = recommend(emp, models)

    if "result" in st.session_state:
        emp, r = st.session_state["employee"], st.session_state["result"]
        pct = r["probability"] * 100
        meter = (f'<div class="meter"><span style="width:{min(pct, 100):.0f}%;'
                 f'background:{STATUS[r["band"]]}"></span></div>')
        st.markdown('<div class="section">Assessment</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        risk_val = f'<span class="count" style="--n:{int(pct)}"></span>.{int(round(pct * 10)) % 10}%'
        k1.markdown(card("Attrition risk", risk_val, extra=band_pill(r["band"]) + meter), unsafe_allow_html=True)
        k2.markdown(card("Segment", r["segment_name"], "K-means cluster", small=True, delay=0.1), unsafe_allow_html=True)
        k3.markdown(card("Pay vs benchmark", f"{r['pay_gap_pct']:+.1f}%",
                         f"Benchmark ${r['income_benchmark']:,.0f} / month", delay=0.2), unsafe_allow_html=True)
        top = r["actions"][0]["driver"] if r["actions"] else "None"
        k4.markdown(card("Main driver", top, f"{len(r['actions'])} actions recommended", small=True, delay=0.3),
                    unsafe_allow_html=True)

        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown('<div class="section">Recommended retention actions</div>', unsafe_allow_html=True)
            if not r["actions"]:
                st.markdown('<div class="hint">No specific risk factors found. Keep up regular check-ins.</div>',
                            unsafe_allow_html=True)
            for i, a in enumerate(r["actions"], 1):
                st.markdown(f'<div class="action" style="animation-delay:{0.25 + 0.1 * i:.2f}s"><div class="num">{i}</div><div>'
                            f'<div class="atext">{a["action"]}</div>'
                            f'<div class="ameta">{a["driver"]} · Owner: {a["owner"]} · '
                            f'Model impact {a["impact"]:+.2f}</div></div></div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section">What is raising the risk</div>', unsafe_allow_html=True)
            st.markdown('<div class="hint">Contribution of each factor to the log-odds of leaving, '
                        'compared with an average employee.</div>', unsafe_allow_html=True)
            if r["top_drivers"]:
                s = pd.Series({pretty(k): v for k, v in r["top_drivers"]})
                st.plotly_chart(hbar(s, unit="", height=280), use_container_width=True,
                                config={"displayModeBar": False})

        st.markdown('<div class="section">Where this employee sits</div>', unsafe_allow_html=True)
        st.markdown('<div class="hint">Drag to rotate, scroll to zoom. Each dot is one employee, '
                    'placed by PCA on the segmentation features.</div>', unsafe_allow_html=True)
        st.plotly_chart(map_3d(emp, r, height=500), use_container_width=True,
                        config={"displayModeBar": False})

# ---------------------------------------------------------------- Tab 2
with tab2:
    wf = score_workforce()
    cur = wf[wf["Attrition"] == "No"]
    b = cur["Band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(card("Current employees", f"{len(cur):,}", "Scored by the model"), unsafe_allow_html=True)
    for i, (col, band) in enumerate(zip([k2, k3, k4], ["Low", "Medium", "High"]), 1):
        col.markdown(card(f"{band} risk", f'<span class="count" style="--n:{b[band]}"></span>',
                          f"{b[band] / len(cur):.0%} of staff", extra=band_pill(band), delay=0.1 * i),
                     unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section">Average risk by department</div>', unsafe_allow_html=True)
        st.plotly_chart(hbar(cur.groupby("Department")["Risk %"].mean()), use_container_width=True,
                        config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="section">Average risk by segment</div>', unsafe_allow_html=True)
        st.plotly_chart(hbar(cur.groupby("Segment")["Risk %"].mean()), use_container_width=True,
                        config={"displayModeBar": False})
    st.markdown('<div class="section">Highest-risk employees</div>', unsafe_allow_html=True)
    f_dept = st.multiselect("Department", cats["Department"], default=cats["Department"])
    view = cur[cur["Department"].isin(f_dept)].sort_values("Risk %", ascending=False)
    st.dataframe(view[["JobRole", "Department", "Age", "MonthlyIncome", "OverTime",
                       "Risk %", "Band", "Segment", "Top action"]].head(25),
                 use_container_width=True, hide_index=True,
                 column_config={"Risk %": st.column_config.ProgressColumn("Risk", format="%.1f%%",
                                                                          min_value=0, max_value=100),
                                "MonthlyIncome": st.column_config.NumberColumn("Income", format="$%d"),
                                "JobRole": "Job role"})

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.markdown('<div class="section">3D workforce map</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hint">All {len(data):,} employees projected onto the first three principal '
                f'components of the segmentation features: {AXES[0].lower()} (experience, income, tenure), '
                f'{AXES[1].lower()}, and {AXES[2].lower()}. Colours are the K-means segments; hover a dot '
                f'for its role and predicted risk. Drag to rotate.</div>', unsafe_allow_html=True)
    if "result" in st.session_state:
        fig3d = map_3d(st.session_state["employee"], st.session_state["result"], height=640, rotate=True)
    else:
        fig3d = map_3d(height=640, rotate=True)
    st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": False})
    seg = pd.DataFrame(results["segments"])
    cols = st.columns(4)
    for i, (col, (_, s)) in enumerate(zip(cols, seg.sort_values("AttritionRate", ascending=False).iterrows())):
        col.markdown(card(s["Name"], f"{s['AttritionRate']:.1f}%", delay=0.1 * i, sub=
                          f"attrition · {int(s['Employees'])} employees · {s['OverTimeShare']:.0f}% overtime",
                          small=True,
                          extra=f'<span class="pill"><span class="dot" style="background:{SEGMENT_COLOURS[s["Name"]]}"></span>segment</span>'),
                     unsafe_allow_html=True)

# ---------------------------------------------------------------- Tab 4
with tab4:
    fig = os.path.join(HERE, "figures")
    if "ensemble" in results:
        ens = results["ensemble"]
        b = ens["best"]
        st.markdown('<div class="section">Deployed model: ensemble</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hint">{b["Model"]}. Chosen on 5-fold cross-validated AUC (repeated twice) '
                    f'on the training set, then scored once on the {ens["n_test"]}-employee hold-out test set '
                    f'({ens["test_leavers"]} leavers). Threshold {b["Threshold"]} from out-of-fold Youden\'s J.</div>',
                    unsafe_allow_html=True)
        ci = b["Test_AUC_CI"]
        cols = st.columns(4)
        for i, (col, (lab, val, sub)) in enumerate(zip(cols, [
                ("Test AUC", f'{b["Test_AUC"]:.3f}', f"95% CI {ci[0]:.2f} – {ci[1]:.2f}"),
                ("Recall", f'{b["Recall"]:.0%}', "leavers caught"),
                ("Precision", f'{b["Precision"]:.0%}', "flags that were right"),
                ("Baseline LR", f'{ens["baseline_test_auc"]:.3f}', "test AUC before")])):
            col.markdown(card(lab, val, delay=0.1 * i, sub=sub, small=True), unsafe_allow_html=True)
        st.markdown(f'<div class="section">{len(ens["experiments"])} model combinations compared</div>', unsafe_allow_html=True)
        exp = pd.DataFrame(ens["experiments"]).drop(columns=["Seconds"]).sort_values("CV_AUC", ascending=False)
        st.dataframe(exp, use_container_width=True, hide_index=True,
                     column_config={"CV_AUC": st.column_config.ProgressColumn("CV AUC", format="%.3f", min_value=0.5, max_value=1),
                                    "CV_SD": "CV SD", "Test_AUC": "Test AUC", "Train_AUC": "Train AUC"})
        c1, c2 = st.columns(2, gap="large")
        c1.image(os.path.join(fig, "17_ensemble_comparison.png"), caption="CV AUC (bars) and test AUC (dots)")
        c2.image(os.path.join(fig, "18_ensemble_roc.png"), caption="ROC on the hold-out test set")
        st.markdown(f'<div class="hint"><b>Why not 0.95?</b> The random forest scores {ens["rf_train_auc"]:.2f} '
                    f'on the data it was trained on but much less on unseen staff: that gap is memorisation, not skill. '
                    f'Careful analyses of this dataset usually report about 0.80 – 0.87 AUC, because it holds only 237 leavers '
                    f'and factors like a job offer elsewhere are not recorded. A 0.95 here would signal leakage.</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="section">Baseline classifiers (DA2)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hint">25% hold-out test set. Logistic regression had the highest AUC of the DA2 models; '
                f'with the ROC-optimal threshold of {results["lr_youden_threshold"]} its recall rises to '
                f'{results["lr_tuned"]["Recall"]:.0%}.</div>', unsafe_allow_html=True)
    perf_df = pd.DataFrame(results["classifier_performance"]).set_index("Model")
    st.dataframe(perf_df, use_container_width=True,
                 column_config={"AUC": st.column_config.ProgressColumn("AUC", format="%.3f", min_value=0.5, max_value=1)})
    c1, c2 = st.columns(2, gap="large")
    c1.image(os.path.join(fig, "08_roc_curves.png"), caption="ROC curves")
    c2.image(os.path.join(fig, "10_lr_coefficients.png"), caption="Logistic regression drivers")
    st.markdown('<div class="section">Significant attrition drivers (chi-square tests)</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(results["chi_square"]), use_container_width=True, hide_index=True)
