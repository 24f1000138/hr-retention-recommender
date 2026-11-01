"""Builds the DA 1, DA 2 and DA 3 Word reports from results.json and figures/."""
import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SHOT = os.path.join(HERE, "screenshots")
OUT = os.path.join(HERE, "reports")
R = json.load(open(os.path.join(HERE, "results.json")))
ENS = json.load(open(os.path.join(HERE, "ensemble_results.json")))
EB = ENS["best"]
os.makedirs(OUT, exist_ok=True)

TITLE = "Employee Retention Recommender: Predicting Attrition Risk and Recommending Targeted Retention Actions"
APP_URL = "[insert your Streamlit Cloud URL here]"
DATA_URL = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
perf = {p["Model"]: p for p in R["classifier_performance"]}
LR = perf["Logistic Regression"]
seg = {s["Name"]: s for s in R["segments"]}
OT = next(s for s in R["segments"] if s["OverTimeShare"] > 90)
sig_chi = [c["Variable"] for c in R["chi_square"] if c["Significant"] == "Yes"]
sig_t = [t["Variable"] for t in R["t_tests"] if t["Significant"] == "Yes"]
ors = {t["Term"]: t for t in R["logit"]["terms"]}
fig_no = [0]
tab_no = [0]


def new_doc(da, subtitle):
    d = Document()
    st = d.styles["Normal"]
    st.font.name = "Calibri"; st.font.size = Pt(11)
    for s in d.sections:
        s.left_margin = s.right_margin = Inches(1)
        s.top_margin = s.bottom_margin = Inches(0.9)
    fig_no[0] = tab_no[0] = 0
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("BMGT107L – Business Analytics"); r.bold = True; r.font.size = Pt(13)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Digital Assignment {da}: {subtitle}"); r.font.size = Pt(12)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    r = p.add_run(TITLE); r.bold = True; r.font.size = Pt(16)
    r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x68)
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.add_run("Name: ____________________    Register No.: ______________\n"
              "Faculty: __________________    Slot: ______    Date: 26 October 2026")
    return d


def h(d, text, level=1):
    d.add_heading(text, level=level)


def para(d, text, bold_lead=None):
    p = d.add_paragraph()
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(6)
    return p


def bullets(d, items, style="List Bullet"):
    for it in items:
        p = d.add_paragraph(style=style)
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hex_fill)
    tcPr.append(sh)


def table(d, rows, cols=None, caption=None, headers=None, font=9):
    if caption:
        tab_no[0] += 1
        c = d.add_paragraph(); c.add_run(f"Table {tab_no[0]}: {caption}").bold = True
    cols = cols or list(rows[0].keys())
    headers = headers or cols
    t = d.add_table(rows=1, cols=len(cols)); t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hd in enumerate(headers):
        cell = t.rows[0].cells[i]; cell.text = ""
        run = cell.paragraphs[0].add_run(str(hd)); run.bold = True; run.font.size = Pt(font)
        shade(cell, "D9E2F3")
    for r in rows:
        cells = t.add_row().cells
        for i, c in enumerate(cols):
            v = r[c]
            if isinstance(v, float):
                v = f"{v:.3g}" if abs(v) < 0.001 and v != 0 else (f"{v:,.2f}" if abs(v) >= 100 else f"{v:.3f}".rstrip("0").rstrip("."))
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(font)
    d.add_paragraph()


def figure(d, path, caption, width=6.0):
    if not os.path.exists(path):
        return
    d.add_picture(path, width=Inches(width))
    d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_no[0] += 1
    c = d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"Figure {fig_no[0]}: {caption}"); r.italic = True; r.font.size = Pt(9)


REFERENCES = [
    "Allen, D. G., Bryant, P. C., & Vardaman, J. M. (2010). Retaining talent: Replacing misconceptions with evidence-based strategies. Academy of Management Perspectives, 24(2), 48–64.",
    "Camm, J. D., Cochran, J. J., Fry, M. J., Ohlmann, J. W., & Anderson, D. R. (2017). Essentials of Business Analytics (2nd ed.). Cengage Learning.",
    "Dinesh Kumar, U. (2017). Business Analytics: The Science of Data-Driven Decision Making (1st ed.). Wiley India.",
    "Evans, J. R. (2021). Business Analytics: Methods, Models and Decisions (3rd ed.). Pearson Education.",
    "Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.). Wiley.",
    f"IBM (2017). IBM HR Analytics Employee Attrition & Performance [Data set]. Kaggle. {DATA_URL}",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. Proceedings of the 9th Python in Science Conference, 92–96.",
    "Shmueli, G., Bruce, P. C., Yahav, I., Patel, N. R., & Lichtendahl, K. C. (2017). Data Mining for Business Analytics: Concepts, Techniques, and Applications in R. Wiley.",
    "Streamlit Inc. (2024). Streamlit documentation. https://docs.streamlit.io",
]

INTRO = (
    "Employee attrition means employees leaving an organisation voluntarily, and it costs a company "
    "a lot. Each resignation brings recruitment and onboarding costs, lost output while a "
    "replacement is hired and trained, and lost knowledge and client relationships. High attrition "
    "also lowers the morale of the people who stay. Much of it comes from things the organisation "
    "can change, such as workload, pay, career growth and job satisfaction (Allen et al., 2010).")
INTRO2 = (
    "Most HR departments react to attrition only after someone resigns. The HR data companies "
    "already collect is enough to act earlier. It can show which employees are likely to leave and "
    "why, so HR can step in before the resignation. This project builds that kind of recommendation "
    "model with the statistical techniques from the Business Analytics syllabus, applied to HR "
    "analytics (Module 6).")
PROBLEM = (
    "HR managers do not know in advance which employees are likely to leave, or which retention "
    "action would suit each of them. So retention money tends to go into company-wide schemes that "
    "miss the people most at risk. The problem is to build a model that (i) predicts each "
    "employee's probability of leaving, (ii) shows which factors drive that risk, and (iii) "
    "recommends retention actions for that employee, in order of priority.")
OBJECTIVES = [
    "To explore and summarise the HR data using descriptive statistics and visualisation.",
    "To test statistically which demographic, job and satisfaction factors are significantly associated with attrition (chi-square tests, t-tests, ANOVA).",
    "To build a salary benchmark using multiple linear regression and flag employees paid well below it.",
    "To build and compare classification models (binary logistic regression, decision tree, KNN, Naive Bayes, LDA) for predicting attrition, evaluated with confusion matrices and ROC/AUC.",
    "To segment employees with K-means and hierarchical clustering, supported by PCA and factor analysis, and profile each segment's attrition risk.",
    "To combine these results into a recommendation engine that suggests targeted retention actions per employee, and deploy it as a live web application.",
]

DATASET_DESC = (
    f"The model uses the IBM HR Analytics Employee Attrition & Performance dataset, a public dataset "
    f"created by IBM data scientists and published on Kaggle ({DATA_URL}). It contains "
    f"{R['n_rows']:,} employee records and 35 variables, with no missing values. The target variable, "
    f"Attrition (Yes/No), shows that {R['attrition_counts']['Yes']} employees "
    f"({R['attrition_rate']}%) left the company and {R['attrition_counts']['No']:,} stayed.")
VAR_GROUPS = [
    {"Group": "Demographic", "Variables": "Age, Gender, MaritalStatus, Education, EducationField, DistanceFromHome", "Scale": "Ratio / nominal / ordinal"},
    {"Group": "Job", "Variables": "Department, JobRole, JobLevel, BusinessTravel, OverTime", "Scale": "Nominal / ordinal"},
    {"Group": "Compensation", "Variables": "MonthlyIncome, PercentSalaryHike, StockOptionLevel, DailyRate, HourlyRate, MonthlyRate", "Scale": "Ratio / ordinal"},
    {"Group": "Experience", "Variables": "TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, NumCompaniesWorked, TrainingTimesLastYear", "Scale": "Ratio"},
    {"Group": "Satisfaction & performance", "Variables": "JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance, JobInvolvement, PerformanceRating (1 to 4 scales)", "Scale": "Ordinal (Likert)"},
    {"Group": "Target", "Variables": "Attrition (Yes / No)", "Scale": "Nominal (binary)"},
]

METHOD_TABLE = [
    {"Stage": "1. Data preparation", "Technique": "Cleaning, removing constant columns, encoding, scaling", "Module": "1"},
    {"Stage": "2. Descriptive analytics", "Technique": "Central tendency, dispersion, histogram, bar, pie, box, scatter plots, correlation", "Module": "2"},
    {"Stage": "3. Hypothesis testing", "Technique": "Chi-square test of independence, Welch t-test, Mann-Whitney U, one-way ANOVA", "Module": "2"},
    {"Stage": "4. Salary benchmark", "Technique": "Simple and multiple linear regression, VIF, residual diagnostics", "Module": "3"},
    {"Stage": "5. Attrition prediction", "Technique": "Binary logistic regression, decision tree, KNN, Naive Bayes, LDA; confusion matrix, ROC/AUC", "Module": "4"},
    {"Stage": "5b. Ensemble learning", "Technique": "Feature engineering, SMOTE, bagging, boosting, soft voting, stacking; repeated k-fold CV, bootstrap CI", "Module": "4 (extension)"},
    {"Stage": "6. Segmentation", "Technique": "PCA, factor analysis, K-means, hierarchical clustering", "Module": "5"},
    {"Stage": "7. Recommendation engine", "Technique": "Risk score, driver contributions and segment combined into ranked actions", "Module": "6"},
    {"Stage": "8. Deployment & report", "Technique": "Streamlit web app; report writing", "Module": "7"},
]


# ======================================================================= DA 1
def build_da1():
    d = new_doc(1, "Problem / Idea of the Model")
    h(d, "1. Title of the Project")
    para(d, TITLE)
    h(d, "2. Introduction and Background")
    para(d, INTRO.replace(" (Allen et al., 2010)", "")); para(d, INTRO2)
    para(d, "Attrition is usually concentrated in certain roles and groups of employees. A first look "
              "at the chosen dataset shows this: about one in six employees left, and the rate is much "
              "higher for people who work overtime, are single, are early in their careers or travel "
              "often. A statistical model can learn these patterns and use them to suggest actions.")
    h(d, "3. Problem Statement")
    para(d, PROBLEM)
    h(d, "4. Objectives of the Model")
    bullets(d, OBJECTIVES, "List Number")
    h(d, "5. Proposed Dataset and Source")
    para(d, DATASET_DESC)
    table(d, VAR_GROUPS, caption="Variables in the dataset")
    para(d, "Four columns carry no information and will be removed: EmployeeCount (always 1), "
              "Over18 (always Y), StandardHours (always 80) and EmployeeNumber (an identifier). The "
              "dataset suits the project because it is public, clean and structured like real HR "
              "records. It has the outcome (attrition) as well as factors a company can act on, such "
              "as overtime, pay, promotion and satisfaction.")
    h(d, "6. Proposed Methodology and Statistical Techniques")
    para(d, "The model will be built in Python (pandas, scipy, statsmodels, scikit-learn) and "
              "deployed as a Streamlit web application. The work moves from description to "
              "prediction and then to recommendations:")
    table(d, METHOD_TABLE, caption="Proposed methodology mapped to syllabus modules")
    bullets(d, [
        ("Descriptive analytics: ", "summarise each variable (mean, median, standard deviation, skewness) and visualise attrition by department, role, overtime and other factors."),
        ("Hypothesis testing: ", "H0: a factor is independent of attrition. Chi-square tests for categorical factors and t-tests for numeric factors at α = 0.05 will identify the significant drivers."),
        ("Regression: ", "multiple linear regression of MonthlyIncome on job level, experience and tenure gives a fair-pay benchmark; employees paid far below it can be flagged."),
        ("Classification: ", "five classifiers will be trained on 75% of the data and compared on the remaining 25% using accuracy, precision, recall, F1 and AUC. The best one becomes the risk-scoring engine."),
        ("Clustering: ", "K-means on experience, pay, satisfaction and overtime will group employees into segments with distinct risk profiles; PCA and factor analysis will reveal the underlying dimensions."),
        ("Recommendation logic: ", "for each employee, the risk score and each factor's contribution will be matched to a library of actions (for example, overtime leads to workload rebalancing, and no stock options leads to a long-term incentive), then ranked by impact."),
    ])
    h(d, "7. Expected Recommendations and Outcomes")
    bullets(d, [
        "An attrition risk score (0 to 100%) and a Low, Medium or High risk band for every employee.",
        "A ranked list of retention actions for each at-risk employee, with who is responsible (line manager, HR or the compensation team).",
        "The statistically significant drivers of attrition across the organisation, such as overtime, low satisfaction, no stock options or slow promotion.",
        "Employee segments, each with its own retention strategy.",
        "A live web application where an HR manager enters an employee's details and gets the risk score and recommendations straight away.",
    ])
    para(d, "For the business, this means HR can spend its retention budget on the employees and "
              "causes where it is most likely to reduce attrition, and act before people resign.")
    d.save(os.path.join(OUT, "DA1_Problem_and_Idea.docx"))


# ======================================================================= DA 2
def implementation_sections(d, final=False):
    """Shared analysis content for DA 2 and DA 3."""
    hn = "4" if not final else "3"
    h(d, ("3. Data Preprocessing" if not final else "3.3 Data Preprocessing"), 1 if not final else 2)
    bullets(d, [
        f"Checked for missing values ({R['missing']} found) and duplicate rows (none).",
        "Removed four constant or identifier columns: EmployeeCount, Over18, StandardHours, EmployeeNumber.",
        "Created a binary target AttritionFlag (Yes = 1) and an OverTimeFlag dummy.",
        "One-hot encoded the seven nominal variables (drop-first) and standardised the 23 numeric variables (z-scores) inside a scikit-learn pipeline, so the same transformation is applied when the app scores a new employee.",
        f"Split the data 75/25 into training ({R['train_n']:,}) and test ({R['test_n']}) sets, stratified on attrition so both sets keep the {R['attrition_rate']}% attrition rate.",
    ])


def descriptive_section(d, lvl=2, prefix="4"):
    h(d, f"{prefix}.1 Descriptive Analytics", lvl)
    para(d, f"The overall attrition rate is {R['attrition_rate']}%, so the classes are imbalanced "
              f"(roughly 5 stayers for every leaver). Table {tab_no[0] + 1} summarises the main numeric variables.")
    table(d, R["descriptive"], cols=["Variable", "mean", "median", "std", "min", "max", "skew"],
          headers=["Variable", "Mean", "Median", "Std dev", "Min", "Max", "Skewness"],
          caption="Descriptive statistics")
    para(d, "MonthlyIncome and YearsAtCompany are strongly right-skewed (mean above median), so "
              "non-parametric Mann-Whitney tests are reported alongside t-tests.")
    figure(d, os.path.join(FIG, "01_attrition_distribution.png"), "Attrition count and share", 5.2)
    figure(d, os.path.join(FIG, "04_attrition_rate_by_category.png"), "Attrition rate by overtime, marital status, travel and job role")
    ot = R["chi_square"][0]
    para(d, "Attrition is about three times higher for employees who work overtime than for those "
              "who do not. It is also highest among Sales Representatives, Laboratory Technicians "
              "and HR staff, single employees, and frequent travellers.")
    figure(d, os.path.join(FIG, "03_boxplots.png"), "Box plots of income, tenure and experience by attrition")
    para(d, "Leavers earn less and have fewer years of experience and tenure than stayers. The "
              "experience and income scatter plot and the correlation heatmap confirm that income, job "
              "level, age and experience are highly inter-correlated, which matters for regression.")
    figure(d, os.path.join(FIG, "05_scatter_experience_income.png"), "Total working years vs monthly income", 4.8)
    figure(d, os.path.join(FIG, "06_correlation_heatmap.png"), "Correlation matrix of numeric variables", 5.5)
    roles = sorted(R["role_table"], key=lambda r: -r["Rate"])[:5]
    table(d, roles, cols=["Department", "JobRole", "Headcount", "Leavers", "Rate"],
          headers=["Department", "Job role", "Headcount", "Leavers", "Attrition %"],
          caption="Job roles with the highest attrition")


def hypothesis_section(d, lvl=2, prefix="4"):
    h(d, f"{prefix}.2 Hypothesis Testing", lvl)
    para(d, "For each categorical factor, H0: the factor is independent of attrition; H1: they are "
              "associated. A chi-square test of independence was used at α = 0.05, with Cramér's V "
              "for effect size.")
    table(d, R["chi_square"], cols=["Variable", "Chi2", "df", "p_value", "Cramers_V", "Significant"],
          headers=["Variable", "χ²", "df", "p-value", "Cramér's V", "Significant"],
          caption="Chi-square tests of independence with attrition")
    para(d, f"{len(sig_chi)} of {len(R['chi_square'])} factors are significant. OverTime and JobRole "
              f"have the strongest association (V ≈ 0.24). Gender is not significant "
              f"(p = {next(c['p_value'] for c in R['chi_square'] if c['Variable'] == 'Gender')}), so "
              f"the recommendations do not depend on gender.")
    para(d, "For numeric factors, H0: mean value is equal for leavers and stayers. Welch's t-test "
              "(unequal variances) and the Mann-Whitney U test were applied.")
    table(d, R["t_tests"], cols=["Variable", "Mean_Leavers", "Mean_Stayers", "t", "p_value_t", "p_value_MW", "Significant"],
          headers=["Variable", "Mean (left)", "Mean (stayed)", "t", "p (t-test)", "p (M-W)", "Significant"],
          caption="Two-sample tests: leavers vs stayers")
    para(d, "Leavers are younger, earn about $2,000 less per month, have less experience, shorter "
              "tenure and fewer years with their manager, and live further from work. Salary hike "
              "and hourly rate do not differ significantly. A one-way ANOVA of monthly income "
              f"across job levels (F = {R['anova_income_joblevel']['F']:,}, p < 0.001) confirms that "
              "job level is the main determinant of pay, which supports the salary benchmark below.")


def regression_section(d, lvl=2, prefix="4"):
    h(d, f"{prefix}.3 Regression: Salary Benchmark", lvl)
    para(d, "To judge whether an employee is fairly paid, monthly income was modelled against job "
              "level, total working years, years at company, age and performance rating.")
    para(d, f"Simple linear regression on JobLevel alone gives R² = {R['slr']['r2']} "
              f"(Income = {R['slr']['coef']['const']:,.0f} + {R['slr']['coef']['JobLevel']:,.0f} × JobLevel). "
              f"The multiple linear regression raises adjusted R² to {R['mlr']['adj_r2']} with a lower "
              f"AIC ({R['mlr']['aic']:,} vs {R['slr']['aic']:,}), so the MLR is preferred "
              f"(F = {R['mlr']['f']:,}, p < 0.001).")
    table(d, R["mlr"]["coef"], cols=["Term", "Coef", "Std_Err", "t", "p_value"],
          headers=["Term", "Coefficient", "Std error", "t", "p-value"],
          caption="Multiple linear regression of MonthlyIncome")
    table(d, R["vif"], caption="Variance inflation factors", headers=["Variable", "VIF"])
    para(d, f"All VIFs are below 5, so multicollinearity is acceptable. The Durbin-Watson statistic "
              f"is {R['mlr']['durbin_watson']} (no autocorrelation). The residual plots show some "
              f"heteroscedasticity at high incomes, which is common for salary data and acceptable for "
              f"a benchmark.")
    figure(d, os.path.join(FIG, "07_regression_diagnostics.png"), "Regression diagnostics")
    para(d, f"Using the benchmark, {R['underpaid_share']}% of employees are paid more than 15% below "
              f"their predicted income. Their attrition rate ({R['attr_underpaid']}%) is "
              f"not higher than for others ({R['attr_not_underpaid']}%; χ² p = {R['underpaid_chi_p']}). "
              f"So being paid below the benchmark is treated as a low-priority pay-equity flag, and low "
              f"absolute income, which is significant, gets a higher priority.")


def classification_section(d, lvl=2, prefix="4"):
    h(d, f"{prefix}.4 Classification: Predicting Attrition", lvl)
    para(d, f"Five classifiers were trained on the same pipeline. KNN's k ({R['knn_k']}) and the "
              f"decision tree's depth ({R['tree_params']['max_depth']}) and minimum leaf size "
              f"({R['tree_params']['min_samples_leaf']}) were chosen by 5-fold cross-validated AUC.")
    table(d, R["classifier_performance"], cols=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "TP", "FP", "FN", "TN"],
          caption="Classifier comparison on the test set (threshold 0.5)")
    figure(d, os.path.join(FIG, "08_roc_curves.png"), "ROC curves of the five classifiers", 4.8)
    figure(d, os.path.join(FIG, "09_confusion_matrices.png"), "Confusion matrices (test set)", 6.3)
    para(d, f"Logistic regression has the highest AUC ({LR['AUC']}), followed by LDA "
              f"({perf['LDA']['AUC']}). Because only 16% of employees leave, accuracy is misleading: "
              f"KNN reaches {perf['KNN']['Accuracy']} accuracy by predicting that nobody leaves. Naive "
              f"Bayes catches more leavers but with many false alarms. Logistic regression was the best "
              f"single classifier and its coefficients are interpretable, so it is kept to explain each "
              f"employee's drivers; Section {prefix}.5 tests whether an ensemble can rank risk better.")
    para(d, f"At the default 0.5 cut-off, logistic regression finds only {LR['Recall']:.0%} of leavers. "
              f"Using the ROC-optimal (Youden's J) threshold of {R['lr_youden_threshold']}, recall rises "
              f"to {R['lr_tuned']['Recall']:.0%} with precision {R['lr_tuned']['Precision']:.0%} and "
              f"accuracy {R['lr_tuned']['Accuracy']:.0%}. The risk bands are set around this point: "
              f"Low below 20%, Medium 20 to 40%, High 40% or more.")
    para(d, f"A statsmodels logit on the main drivers (pseudo R² = {R['logit']['pseudo_r2']}, "
              f"likelihood-ratio p < 0.001) gives interpretable odds ratios:")
    keep = ["OverTimeFlag", "TravelFrequently", "Single", "JobInvolvement", "EnvironmentSatisfaction",
            "JobSatisfaction", "WorkLifeBalance", "YearsSinceLastPromotion", "NumCompaniesWorked",
            "DistanceFromHome", "StockOptionLevel", "MonthlyIncome_k"]
    table(d, [ors[k] for k in keep], cols=["Term", "Coef", "Odds_Ratio", "p_value"],
          headers=["Factor", "Coefficient", "Odds ratio", "p-value"], caption="Logistic regression odds ratios")
    para(d, f"Holding other factors constant, overtime multiplies the odds of leaving by "
              f"{ors['OverTimeFlag']['Odds_Ratio']:.1f}, frequent travel by {ors['TravelFrequently']['Odds_Ratio']:.1f} "
              f"and being single by {ors['Single']['Odds_Ratio']:.1f}. Each one-point rise in job "
              f"involvement, environment satisfaction or job satisfaction cuts the odds by about "
              f"{(1 - ors['JobInvolvement']['Odds_Ratio']):.0%}, {(1 - ors['EnvironmentSatisfaction']['Odds_Ratio']):.0%} "
              f"and {(1 - ors['JobSatisfaction']['Odds_Ratio']):.0%} respectively. Each extra year since "
              f"the last promotion raises the odds by {(ors['YearsSinceLastPromotion']['Odds_Ratio'] - 1):.0%}.")
    figure(d, os.path.join(FIG, "10_lr_coefficients.png"), "Largest standardised logistic regression coefficients", 5.0)
    figure(d, os.path.join(FIG, "11_decision_tree.png"), "Decision tree (top three levels)", 6.3)
    para(d, "The decision tree tells the same story in rule form. Its first split separates employees "
              "with very little total experience (about two years or less), and the next split on both "
              "branches is overtime. Among inexperienced employees who work overtime, about 81% left. "
              "Further splits use the number of previous employers, monthly income, work-life balance "
              "and environment satisfaction.")


def ensemble_section(d, lvl=2, prefix="4"):
    h(d, f"{prefix}.5 Ensemble Models: Bagging, Boosting, Voting and Stacking", lvl)
    para(d, f"To test whether a more complex model predicts attrition better, {len(ENS['experiments'])} model combinations were "
              f"compared: single models (logistic regression, SVM), class-imbalance treatments (class "
              f"weights, SMOTE), bagging (random forest, extra trees), boosting (gradient boosting, XGBoost, "
              f"LightGBM, CatBoost) and ensembles (soft voting, weighted voting and stacking). Seven "
              f"engineered features were added: income per job level, tenure ratio, promotion gap ratio, "
              f"role and manager tenure ratios, job-hop rate and a satisfaction index.")
    para(d, f"Every model was evaluated with repeated stratified 5-fold cross-validation (2 repeats) on "
              f"the {ENS['n_train']:,} training employees, and then scored once on the untouched hold-out "
              f"test set of {ENS['n_test']} employees ({ENS['test_leavers']} leavers). The deployed model "
              f"was chosen on cross-validated AUC only, never on the test set, so the test score is an "
              f"honest estimate. SMOTE was applied inside each training fold to avoid leakage.")
    rows = sorted(ENS["experiments"], key=lambda r: -r["CV_AUC"])
    table(d, rows, cols=["Model", "Family", "CV_AUC", "CV_SD", "Test_AUC", "Train_AUC"],
          headers=["Model", "Family", "CV AUC", "CV SD", "Test AUC", "Train AUC"],
          caption=f"{len(ENS['experiments'])} model combinations (CV on training set, then hold-out test)", font=8)
    figure(d, os.path.join(FIG, "17_ensemble_comparison.png"), "CV AUC (bars, ± 1 SD) and test AUC (dots) for each combination", 6.0)
    para(d, f"The selected model is the {EB['Model']}. On the hold-out test set it reaches AUC "
              f"{EB['Test_AUC']} (bootstrap 95% CI {EB['Test_AUC_CI'][0]} to {EB['Test_AUC_CI'][1]}), compared "
              f"with {ENS['baseline_test_auc']} for the original logistic regression. With the threshold of "
              f"{EB['Threshold']} chosen by Youden's J on out-of-fold training predictions, it catches "
              f"{EB['Recall']:.0%} of leavers with {EB['Precision']:.0%} precision (accuracy "
              f"{EB['Accuracy']:.0%}, F1 {EB['F1']}).")
    figure(d, os.path.join(FIG, "18_ensemble_roc.png"), "ROC curves on the hold-out test set", 4.8)
    para(d, f"Why the scores do not reach 0.95. The tree models show what happens when a model is judged "
              f"on its training data. The random forest scores AUC {ENS['rf_train_auc']:.2f} on the employees it was trained "
              f"on but far less on unseen employees. It has memorised the training data. The dataset "
              f"has only 237 leavers, and much of what makes a person resign (an outside offer, a manager "
              f"change, personal events) is not recorded, so some uncertainty cannot be removed. Careful "
              f"analyses of this dataset typically report test AUC of roughly 0.80 to 0.87. A score above 0.95 "
              f"would therefore suggest leakage, such as scoring on training data or tuning on the test set. "
              f"The ensemble has the highest cross-validated AUC of all combinations and blends models that make "
              f"different kinds of errors, but its honest gain over logistic regression is small and lies "
              f"inside the confidence interval. Ensembles that leaned on deep trees (random forest, CatBoost) "
              f"did worse on the test set because they overfit. Logistic regression is still used to explain "
              f"each employee's drivers.")


def clustering_section(d, lvl=2, prefix="4"):
    h(d, f"{prefix}.6 Dimensionality Reduction and Clustering", lvl)
    para(d, f"PCA on the 23 standardised numeric variables shows that the first component explains "
              f"{R['pca_var'][0]}% of variance and loads on {', '.join(R['pca_top_loadings']['PC1'][:4])}: a "
              f"'seniority' dimension. {R['pca_kaiser']} components have eigenvalues above 1 and "
              f"{R['pca_n_80']} are needed for 80% of variance, so the remaining variables are fairly "
              f"independent. Factor analysis (3 factors, varimax) gives a similar picture: "
              f"F1 = seniority/pay ({', '.join(R['fa_top_loadings']['F1'])}), F2 = tenure "
              f"({', '.join(R['fa_top_loadings']['F2'][:3])}), F3 = pay rise/performance.")
    figure(d, os.path.join(FIG, "12_pca.png"), "PCA scree plot and employees on the first two components")
    k_best = max(R["silhouette"], key=lambda k: R["silhouette"][k])
    para(d, f"K-means was run on nine features (age, income, experience, tenure, years since "
              f"promotion, three satisfaction scores and overtime). The silhouette score peaks at "
              f"k = {k_best} ({R['silhouette'][k_best]}), but k = 4 was chosen because it is close to the "
              f"elbow, has a similar silhouette to k = 3, and gives segments that are more useful for HR.")
    figure(d, os.path.join(FIG, "13_kmeans_selection.png"), "Elbow and silhouette method for choosing k")
    table(d, R["segments"], cols=["Name", "Employees", "AttritionRate", "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "OverTimeShare"],
          headers=["Segment", "Employees", "Attrition %", "Avg age", "Avg income", "Avg tenure", "Job sat.", "Overtime %"],
          caption="K-means segment profiles (k = 4)")
    para(d, f"Segment attrition differs significantly (χ² p = {R['segment_chi_p']}). The "
              f"'{OT['Name']}' segment ({OT['Employees']} employees, all working overtime) has "
              f"{OT['AttritionRate']:.1f}% attrition, about three times that of the other segments. "
              f"The senior segments have the lowest attrition. A Ward hierarchical clustering on a "
              f"sample (dendrogram below) supports a solution of three to four groups.")
    figure(d, os.path.join(FIG, "14_segment_attrition.png"), "Attrition rate by segment", 4.8)
    figure(d, os.path.join(FIG, "15_dendrogram.png"), "Hierarchical clustering dendrogram")


def engine_section(d, lvl=2, prefix="4", num=7):
    h(d, f"{prefix}.{num} Recommendation Engine", lvl)
    para(d, "The recommendation engine combines the three models for each employee:")
    bullets(d, [
        ("Risk score: ", f"the probability of leaving from the deployed ensemble ({EB['Model'].split(': ')[1]}), mapped to Low / Medium / High."),
        ("Driver contributions: ", "from the logistic regression: for each factor, coefficient × (employee's standardised value − average), showing how much that factor pushes this employee's log-odds above average."),
        ("Pay check: ", "actual income vs the MLR benchmark."),
        ("Segment: ", "the employee's K-means segment."),
        ("Action library: ", "14 rules, each linking a statistically significant driver to a retention action and an owner. A rule fires only if the employee has that risk factor. Fired actions are ranked by priority weight + 2 × the model contribution of the related factors."),
        ("Output size: ", "Low-risk employees get at most 2 actions, Medium 4 and High 6, so most effort goes to the highest-risk people."),
    ])
    rules = [
        ("Regular overtime", "Rebalance workload, cap overtime", "Line manager"),
        ("Low absolute pay (< $3,000)", "Market pay review / targeted increment", "HR / Compensation"),
        ("Low job / environment satisfaction", "Stay interview, role enrichment, fix workplace issues", "Manager / HRBP"),
        ("Poor work-life balance", "Flexible hours or hybrid days", "Line manager"),
        ("Low job involvement", "Clear goals, involvement and recognition", "Line manager"),
        ("No stock options", "Include in stock option / long-term incentive plan", "HR / Compensation"),
        ("Overdue promotion (≥ 4 years)", "Career-path discussion, promotion review", "HR / Talent"),
        ("Frequent business travel", "Reduce travel or compensate", "Line manager"),
        ("Single, entry-level", "Mentor and 12-month development plan", "HR / Talent"),
        ("Long commute (≥ 15 km)", "Remote days or transport allowance", "HR"),
    ]
    table(d, [{"Driver": a, "Action": b, "Owner": c} for a, b, c in rules], caption="Main rules in the action library")


def build_da2():
    d = new_doc(2, "Software / Model Implementation")
    h(d, "1. Brief Introduction and Objectives")
    para(d, INTRO2)
    para(d, "Objectives of the implemented model:")
    bullets(d, OBJECTIVES, "List Number")
    h(d, "2. Dataset Description and Source")
    para(d, DATASET_DESC)
    table(d, VAR_GROUPS, caption="Variables in the dataset")
    implementation_sections(d)
    h(d, "4. Model Methodology and Implementation")
    para(d, "The model is implemented in Python 3 with pandas, scipy, statsmodels, scikit-learn, "
              "matplotlib/seaborn and Streamlit. The code is organised in three files:")
    bullets(d, [
        ("analysis.py: ", "runs all statistical analysis, trains and compares the models, and saves the figures, results.json and the trained model (model.joblib)."),
        ("recommender.py: ", "shared logic: data cleaning, the action library and the function that turns an employee record into a risk score and ranked recommendations."),
        ("app.py: ", "the Streamlit web application used by HR managers."),
    ])
    table(d, METHOD_TABLE, caption="Methodology and syllabus mapping")
    h(d, "5. Statistical Techniques Used")
    descriptive_section(d, 2, "5"); hypothesis_section(d, 2, "5"); regression_section(d, 2, "5")
    classification_section(d, 2, "5"); ensemble_section(d, 2, "5"); clustering_section(d, 2, "5"); engine_section(d, 2, "5")
    h(d, "6. Screenshots and Sample Outputs")
    figure(d, os.path.join(SHOT, "app_01_form.png"), "App: employee profile input form", 6.0)
    figure(d, os.path.join(SHOT, "app_02a_risk.png"), "App: risk score, segment, pay check and recommended actions for a sample employee", 6.0)
    figure(d, os.path.join(SHOT, "app_03_workforce.png"), "App: workforce overview with risk bands", 6.0)
    figure(d, os.path.join(SHOT, "app_05_3d_map.png"), "App: interactive 3D workforce map (PCA, coloured by K-means segment)", 6.0)
    figure(d, os.path.join(SHOT, "app_04_insights.png"), "App: model insights tab with the deployed ensemble", 6.0)
    figure(d, os.path.join(SHOT, "app_06_ensemble.png"), "App: comparison of all model combinations", 6.0)
    para(d, "Sample output for the five current employees with the highest predicted risk:")
    rows = [{"Role": s["JobRole"], "Age": s["Age"], "Risk %": s["Risk"], "Segment": s["Segment"],
             "Top drivers": "; ".join(s["Drivers"])} for s in R["sample_recommendations"]]
    table(d, rows, caption="Sample recommendations (highest-risk current employees)")
    h(d, "7. Initial Results and Recommendations")
    bc = R["band_counts"]
    para(d, f"Scoring the {sum(bc.values()):,} current employees gives {bc['High']} High-risk, "
              f"{bc['Medium']} Medium-risk and {bc['Low']:,} Low-risk employees. Sales has the highest "
              f"average risk ({R['dept_risk']['Sales']}%).")
    figure(d, os.path.join(FIG, "16_action_frequency.png"), "Most common drivers among Medium/High-risk employees", 5.2)
    bullets(d, [
        "Cut and monitor overtime: it is the strongest single driver (odds ratio ≈ 5.6), and the overtime-heavy segment has about 34% attrition.",
        "Extend stock options to employees at level 0, one of the two most common drivers among at-risk staff.",
        "Run stay interviews for employees with low job or environment satisfaction or low involvement.",
        "Give early-career, single, entry-level staff mentors and development plans, especially Sales Representatives and Laboratory Technicians.",
        "Reduce frequent travel and review promotion timelines for staff overdue for promotion.",
    ])
    d.save(os.path.join(OUT, "DA2_Model_Implementation.docx"))


# ======================================================================= DA 3
def build_da3():
    d = new_doc(3, "Final Report and Results")
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Live application: {APP_URL}"); r.bold = True
    h(d, "1. Introduction and Problem Statement")
    para(d, INTRO); para(d, INTRO2)
    para(d, PROBLEM, "Problem statement. ")
    h(d, "2. Objectives")
    bullets(d, OBJECTIVES, "List Number")
    h(d, "3. Dataset Description and Methodology")
    h(d, "3.1 Dataset", 2)
    para(d, DATASET_DESC)
    table(d, VAR_GROUPS, caption="Variables in the dataset")
    h(d, "3.2 Methodology", 2)
    table(d, METHOD_TABLE, caption="Methodology and syllabus mapping")
    implementation_sections(d, final=True)
    h(d, "4. Statistical Analysis and Interpretation of Results")
    descriptive_section(d, 2, "4"); hypothesis_section(d, 2, "4"); regression_section(d, 2, "4")
    classification_section(d, 2, "4"); ensemble_section(d, 2, "4"); clustering_section(d, 2, "4")
    h(d, "5. Model Outputs and Recommendations")
    engine_section(d, 2, "5", num=1)
    h(d, "5.2 Deployed Application", 2)
    para(d, f"The model is deployed as a Streamlit web application at {APP_URL}. It has four tabs: "
              f"(1) Assess an employee: enter a profile and get the risk score, band, segment, pay check, "
              f"ranked actions, a chart of the factors raising risk, and the employee's position on a 3D "
              f"map; (2) Workforce overview: risk bands, risk by department and segment, and a filterable "
              f"list of the highest-risk employees with their top action; (3) 3D workforce map: all "
              f"employees plotted on the first three principal components of the segmentation features "
              f"(seniority, overtime and environment, work-life balance) and coloured by K-means segment, "
              f"which can be rotated and hovered; (4) Model insights: the ensemble results, the comparison of all "
              f"model combinations, the baseline classifiers, ROC curves and significant drivers.")
    figure(d, os.path.join(SHOT, "app_01_form.png"), "Employee profile form", 6.0)
    figure(d, os.path.join(SHOT, "app_02a_risk.png"), "Risk score and recommended actions", 6.0)
    figure(d, os.path.join(SHOT, "app_03_workforce.png"), "Workforce overview", 6.0)
    figure(d, os.path.join(SHOT, "app_05_3d_map.png"), "3D workforce map (PCA components, coloured by segment)", 6.0)
    figure(d, os.path.join(SHOT, "app_04_insights.png"), "Model insights: the deployed ensemble", 6.0)
    h(d, "5.3 Individual-level Outputs", 2)
    for s in R["sample_recommendations"][:3]:
        para(d, f"{s['JobRole']}, age {s['Age']}: risk {s['Risk']}% ({s['Band']}), segment "
                  f"'{s['Segment']}', pay {s['PayGap']:+.1f}% vs benchmark. Recommended: "
                  + " ".join(f"({i}) {a}" for i, a in enumerate(s["Actions"], 1)),
             "Example. ")
    h(d, "5.4 Organisation-level Recommendations", 2)
    figure(d, os.path.join(FIG, "16_action_frequency.png"), "Most common drivers among Medium/High-risk employees", 5.2)
    bullets(d, [
        ("Control overtime. ", f"Overtime multiplies the odds of leaving by {ors['OverTimeFlag']['Odds_Ratio']:.1f}, and the overtime-heavy segment loses {OT['AttritionRate']:.0f}% of staff. Set overtime caps, hire into chronically stretched teams and track overtime as an HR risk indicator."),
        ("Widen stock option coverage. ", "Employees with no stock options leave far more often (χ² p < 0.001), and it is one of the two most common drivers among at-risk staff. Even level-1 options for the highest-risk employees would target this driver directly."),
        ("Act on satisfaction and involvement. ", "Low job satisfaction, environment satisfaction and involvement are each significant. Use pulse surveys to identify these employees and follow up with stay interviews and role enrichment."),
        ("Support early-career staff. ", "Young, single, entry-level employees, especially Sales Representatives and Laboratory Technicians, have the highest attrition. They should get mentoring, a clear 12-month growth plan and early recognition."),
        ("Reduce travel and review promotions. ", f"Frequent travel (OR ≈ {ors['TravelFrequently']['Odds_Ratio']:.1f}) and each additional year since promotion (+{(ors['YearsSinceLastPromotion']['Odds_Ratio'] - 1):.0%} odds) raise risk."),
        ("Focus on Sales. ", f"Sales has the highest average predicted risk ({R['dept_risk']['Sales']}%), so it should be the pilot department for the programme."),
    ])
    h(d, "6. Conclusion and Future Scope")
    para(d, "The syllabus techniques fit together into a working HR tool. Descriptive analysis and "
              "hypothesis tests found the drivers, and regression gave a fair-pay benchmark. "
              f"Classification and ensemble learning gave a risk score with a test AUC of {EB['Test_AUC']}, "
              "chosen by cross-validation. Clustering split employees into segments with very different "
              "risk, and the recommendation engine turns all of this into ranked actions that an HR "
              "manager can use in the web app. The main finding is that the biggest drivers of attrition "
              "(overtime, stock options, satisfaction and promotion) are things management controls.")
    para(d, "The work has limits. The data is one fictional but realistic IBM sample with a 16% "
              "attrition rate, so precision for leavers is moderate. It is a snapshot without dates, so it "
              "cannot prove that an action causes people to stay. The action library comes from the "
              "statistical drivers and HR literature, and the effect of each action has not been measured.")
    para(d, "Future scope:")
    bullets(d, [
        "Retrain on the organisation's own HR data and refresh scores monthly.",
        "Add richer data such as engagement surveys, manager changes and market pay. Tuning the model further gains little on this dataset.",
        "Add survival analysis to predict when an employee is likely to leave as well as whether.",
        "Record which recommended actions were taken and test their effect with A/B tests, then use the results to re-rank the actions.",
        "Add exit-interview and survey text using sentiment analysis, and integrate with the HRMS.",
    ])
    h(d, "References")
    bullets(d, REFERENCES)
    d.save(os.path.join(OUT, "DA3_Final_Report.docx"))


build_da1(); build_da2(); build_da3()
print("done", os.listdir(OUT))
