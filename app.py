import os, calendar
import streamlit as st
import pandas as pd, numpy as np, joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (roc_curve, confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score,
                             roc_auc_score, average_precision_score)
from features import load_clean, load_health_clean, find_data_file, FEATURES

st.set_page_config(page_title="Diabetes Readmission", page_icon="🏥", layout="wide")

# ---------- COLOURS ----------
ORANGE, TEAL, RED, BLUE = "#FF6B35", "#2EC4B6", "#E63946", "#3A86FF"
PALETTE = [ORANGE, TEAL, BLUE]

# ---------- STYLING ----------
st.markdown("""
<style>
.banner {
    background: linear-gradient(90deg, #FF6B35 0%, #F7931E 50%, #2EC4B6 100%);
    padding: 22px 28px; border-radius: 16px; margin-bottom: 22px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}
.banner h1 { margin: 0; color: white; font-size: 2rem; padding: 0; }
.banner p  { margin: 4px 0 0 0; color: rgba(255,255,255,0.92); font-size: 1rem; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1F2E, #252B3D);
    border-left: 5px solid #FF6B35; padding: 14px 18px; border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
}
div[data-testid="stMetricValue"] { color: #FF6B35; font-weight: 700; }
h2, h3 { color: #2EC4B6; }
</style>
""", unsafe_allow_html=True)

def banner(title, subtitle):
    st.markdown(f'<div class="banner"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)

def show(fig, height=520):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=0, r=0, t=30, b=0))
    fig.update_scenes(bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)
    st.caption("Drag to rotate, scroll to zoom, double-click to reset the view.")

# ---------- 3D BAR CHART HELPER ----------
def bars3d(xlabels, ylabels, Z, zlabel, base=0.0, diverging=False,
           text_fmt="{:.1f}", xtitle="", ytitle=""):
    """Real 3D bars. Z has shape (len(ylabels), len(xlabels))."""
    Z = np.atleast_2d(np.array(Z, dtype=float))
    xlabels, ylabels = [str(x) for x in xlabels], [str(y) for y in ylabels]
    nx, ny = len(xlabels), len(ylabels)
    w = 0.38
    tri_i = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
    tri_j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    tri_k = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]

    X, Y, Zs, I, J, K, inten = [], [], [], [], [], [], []
    tx, ty, tz, tt, hv = [], [], [], [], []
    n = 0
    for yi in range(ny):
        for xi in range(nx):
            v = Z[yi, xi]
            z0, z1 = sorted([base, v])
            x0, x1, y0, y1 = xi - w, xi + w, yi - w, yi + w
            X += [x0, x1, x1, x0, x0, x1, x1, x0]
            Y += [y0, y0, y1, y1, y0, y0, y1, y1]
            Zs += [z0] * 4 + [z1] * 4
            I += [t + 8 * n for t in tri_i]
            J += [t + 8 * n for t in tri_j]
            K += [t + 8 * n for t in tri_k]
            inten += [v] * 8
            n += 1
            tx.append(xi); ty.append(yi); tz.append(z1)
            tt.append(text_fmt.format(v))
            hv.append(f"{xlabels[xi]} / {ylabels[yi]}: {text_fmt.format(v)}"
                      if ny > 1 else f"{xlabels[xi]}: {text_fmt.format(v)}")

    vmin, vmax = float(Z.min()), float(Z.max())
    if diverging:
        m = max(abs(vmin), abs(vmax), 1e-9)
        cscale, cmin, cmax = [[0, TEAL], [0.5, "#9CA3AF"], [1, RED]], -m, m
    else:
        cscale, cmin, cmax = [[0, TEAL], [0.5, ORANGE], [1, RED]], vmin, max(vmax, vmin + 1e-9)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=X, y=Y, z=Zs, i=I, j=J, k=K, intensity=inten,
                            colorscale=cscale, cmin=cmin, cmax=cmax,
                            showscale=False, flatshading=True, hoverinfo="skip",
                            lighting=dict(ambient=0.65, diffuse=0.85, specular=0.2)))
    fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode="text", text=tt,
                               textfont=dict(color="white", size=12),
                               hovertext=hv, hoverinfo="text", showlegend=False))

    lo, hi = min(base, vmin), max(base, vmax)
    pad = (hi - lo) * 0.18 or 1.0
    zrange = [lo - (pad if lo < base else 0), hi + pad]
    yaxis = dict(title=ytitle, tickvals=list(range(ny)), ticktext=ylabels)
    if ny == 1:
        yaxis = dict(visible=False)
    fig.update_layout(scene=dict(
        xaxis=dict(title=xtitle, tickvals=list(range(nx)), ticktext=xlabels),
        yaxis=yaxis,
        zaxis=dict(title=zlabel, range=zrange),
        aspectmode="manual",
        aspectratio=dict(x=1.6, y=0.5, z=0.8) if ny == 1 else dict(x=1.4, y=1.0, z=0.8),
        camera=dict(eye=dict(x=1.6, y=-1.8, z=0.9))))
    return fig

# ---------- DATA + MODEL ----------
@st.cache_data
def load_data():
    return load_clean()

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

df = load_data()
bundle = load_model()
model = bundle["model"]
THRESHOLD = bundle["threshold"]
DEFAULTS = bundle["defaults"]
y_test = bundle["y_test"]
prob_test = bundle["prob_test"]

# Validation results copied from the notebook (model comparison table)
VALIDATION = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
    "Accuracy": [0.6475, 0.7981, 0.8861, 0.8861],
    "Precision": [0.1658, 0.1445, 1.0000, 0.5294],
    "Recall": [0.5189, 0.1566, 0.0005, 0.0048],
    "F1": [0.2513, 0.1503, 0.0011, 0.0095],
    "ROC-AUC": [0.6345, 0.5187, 0.6231, 0.6396],
    "PR-AUC": [0.2026, 0.1188, 0.1824, 0.2047],
})

# ---------- PAGE: OVERVIEW ----------
def overview():
    banner("Diabetes 30-Day Readmission Dashboard",
           "UCI Diabetes 130-US Hospitals (1999-2008)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total admissions", f"{len(df):,}")
    c2.metric("30-day readmission", f"{df['readmitted_30'].mean()*100:.1f}%")
    c3.metric("Avg stay (days)", f"{df['time_in_hospital'].mean():.1f}")
    c4.metric("Avg medications", f"{df['num_medications'].mean():.1f}")
    c5.metric("HbA1c tested", f"{df['hba1c_tested'].mean()*100:.1f}%")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Readmission rate by age band")
        age_rate = df.groupby("age")["readmitted_30"].mean() * 100
        show(bars3d(age_rate.index, [""], [age_rate.values],
                    "Readmitted <30 (%)", xtitle="Age band"))
    with right:
        st.subheader("Readmission split")
        split = df["readmitted"].value_counts(normalize=True) * 100
        show(bars3d(split.index, [""], [split.values], "% of admissions",
                    xtitle="Readmitted"))
    st.caption(f"Final model (Gradient Boosting) test ROC-AUC: "
               f"{roc_auc_score(y_test, prob_test):.3f}")

# ---------- PAGE: EDA ----------
def eda():
    banner("Exploratory Data Analysis", "What drives 30-day readmission?")

    st.subheader("Readmission distribution")
    counts = df["readmitted"].value_counts()
    show(bars3d(counts.index, [""], [counts.values], "Admissions",
                text_fmt="{:,.0f}", xtitle="Readmitted"))

    st.subheader("Readmission rate by prior inpatient visits")
    g = df.assign(k=df["number_inpatient"].clip(upper=8)).groupby("k")["readmitted_30"]
    rate = g.mean() * 100
    xl = [str(int(k)) if k < 8 else "8+" for k in rate.index]
    show(bars3d(xl, [""], [rate.values], "Readmitted <30 (%)",
                xtitle="Prior inpatient visits"))
    st.caption("Prior inpatient visits is the strongest predictor in the final model.")

    st.subheader("Hospital stay vs readmission")
    ct = pd.crosstab(df["readmitted"], df["time_in_hospital"], normalize="index") * 100
    show(bars3d(ct.columns, ct.index, ct.values, "% of group",
                text_fmt="{:.0f}", xtitle="Days in hospital", ytitle="Readmitted"), 560)
    st.caption("Each readmission group adds up to 100%, so the groups can be compared fairly.")

    st.subheader("Prior visits vs readmission")
    col = st.selectbox("Choose a variable",
                       ["number_outpatient", "number_emergency", "number_inpatient"])
    ct = pd.crosstab(df["readmitted"], df[col].clip(upper=5), normalize="index") * 100
    xl = [str(c) if c < 5 else "5+" for c in ct.columns]
    show(bars3d(xl, ct.index, ct.values, "% of group", text_fmt="{:.0f}",
                xtitle="Number of visits", ytitle="Readmitted"), 560)

    st.subheader("Correlation surface")
    nums = ["time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses"]
    corr = df[nums].corr()
    n = len(nums)
    fig = go.Figure(go.Surface(z=corr.values, x=list(range(n)), y=list(range(n)),
                               colorscale="RdBu_r", cmin=-1, cmax=1,
                               colorbar=dict(title="Corr")))
    fig.update_layout(scene=dict(
        xaxis=dict(tickvals=list(range(n)), ticktext=nums, title=""),
        yaxis=dict(tickvals=list(range(n)), ticktext=nums, title=""),
        zaxis=dict(title="Correlation", range=[-1, 1]),
        camera=dict(eye=dict(x=1.7, y=-1.7, z=1.0))))
    show(fig, 620)

    st.subheader("Readmission rate by primary diagnosis family")
    fam = (df.groupby("diag_1_group")["readmitted_30"]
           .agg(admissions="size", rate_pct=lambda s: round(s.mean() * 100, 1))
           .reset_index().sort_values("rate_pct"))
    show(bars3d(fam["diag_1_group"], [""], [fam["rate_pct"].values],
                "Readmitted <30 (%)", xtitle="Primary diagnosis family"))
    st.dataframe(fam.sort_values("rate_pct", ascending=False), hide_index=True)
    st.caption("Small families have noisy rates, so check the admissions column.")

# ---------- PAGE: RISK ESTIMATOR ----------
def estimator():
    banner("30-Day Readmission Risk Estimator",
           "Final model: Gradient Boosting (same as the notebook)")
    ADM = {"Emergency": 1, "Urgent": 2, "Elective": 3}
    ages = sorted(df["age"].unique())
    races = sorted(df["race"].unique())
    fams = sorted(df["diag_1_group"].unique())

    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        age = c1.selectbox("Age band", ages, index=ages.index(DEFAULTS["age"]))
        gender = c1.selectbox("Gender", ["Female", "Male"])
        race = c1.selectbox("Race", races, index=races.index(DEFAULTS["race"]))
        adm = c1.selectbox("Admission type", list(ADM))
        diag1 = c1.selectbox("Primary diagnosis family", fams,
                             index=fams.index(DEFAULTS["diag_1_group"]))
        stay = c2.slider("Days in hospital", 1, 14, 4)
        labs = c2.slider("Lab procedures", 1, 130, 43)
        procs = c2.slider("Procedures", 0, 6, 1)
        meds = c2.slider("Number of medications", 1, 80, 16)
        diags = c2.slider("Number of diagnoses", 1, 16, 8)
        inp = c3.number_input("Prior inpatient visits", 0, 20, 0)
        emer = c3.number_input("Prior emergency visits", 0, 20, 0)
        out = c3.number_input("Prior outpatient visits", 0, 20, 0)
        change = c3.selectbox("Medication changed", ["No", "Ch"])
        diabetesMed = c3.selectbox("On diabetes medication", ["Yes", "No"])
        insulin = c3.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        submitted = st.form_submit_button("Estimate risk")

    if not submitted:
        st.info("Fill in the patient profile and click **Estimate risk**. Fields not "
                "listed here (other medications, payer, specialty, secondary diagnoses) "
                "are set to the typical value in the training data.")
        return

    inputs = {
        "age": age, "gender": gender, "race": race,
        "admission_type_id": ADM[adm], "diag_1_group": diag1,
        "time_in_hospital": stay, "num_lab_procedures": labs,
        "num_procedures": procs, "num_medications": meds,
        "number_diagnoses": diags, "number_inpatient": inp,
        "number_emergency": emer, "number_outpatient": out,
        "change": change, "diabetesMed": diabetesMed, "insulin": insulin,
    }
    row = pd.DataFrame([{**DEFAULTS, **inputs}])[FEATURES]
    p = model.predict_proba(row)[0, 1]
    avg = prob_test.mean() * 100

    # what-if: how much does each input move the risk vs a typical value?
    alts = []
    for f in inputs:
        alt = row.copy()
        alt[f] = DEFAULTS[f]
        alts.append(alt)
    p_alt = model.predict_proba(pd.concat(alts, ignore_index=True))[:, 1]
    contrib = pd.Series((p - p_alt) * 100, index=list(inputs))
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(3)

    st.metric("Estimated 30-day readmission risk", f"{p*100:.1f}%")
    if p >= THRESHOLD:
        st.warning(f"Above the model's decision threshold ({THRESHOLD*100:.1f}%): "
                   "this profile would be flagged as higher risk.")
    else:
        st.success(f"Below the model's decision threshold ({THRESHOLD*100:.1f}%): "
                   "this profile would not be flagged.")

    left, right = st.columns(2)
    with left:
        st.subheader("This patient vs threshold vs average")
        show(bars3d(["This patient", "Decision threshold", "Average encounter"],
                    [""], [[p * 100, THRESHOLD * 100, avg]], "Risk (%)"), 440)
    with right:
        st.subheader("Top 3 factors for this patient")
        show(bars3d(top.index, [""], [top.values], "Change in risk (pct points)",
                    diverging=True, text_fmt="{:+.1f}"), 440)
        st.caption("Red raises the risk, teal lowers it, compared with a typical "
                   "value for that input.")
    st.caption("Educational project only, not for clinical use.")

# ---------- PAGE: MODEL COMPARISON ----------
def comparison():
    banner("Model Comparison", "Four models on the validation set (from the notebook)")
    metrics = ["ROC-AUC", "PR-AUC", "F1"]
    show(bars3d(VALIDATION["Model"], metrics, VALIDATION[metrics].T.values,
                "Score", text_fmt="{:.3f}", xtitle="Model", ytitle="Metric"), 560)
    st.caption("F1 is at the default 0.5 threshold, which is why Random Forest and "
               "Gradient Boosting look weak on that bar. Threshold tuning fixes this.")
    st.dataframe(VALIDATION, hide_index=True)
    st.write("Gradient Boosting was selected after tuning the threshold on the "
             "validation set (best F1 0.258 at 0.1176). Logistic Regression reached "
             "0.254 at 0.481, so the two are close.")

# ---------- PAGE: MODEL EVALUATION ----------
def evaluation():
    banner("Model Evaluation", "Final Gradient Boosting model on the held-out test set")

    st.subheader("3D ROC curve (threshold as the third axis)")
    fpr, tpr, thr = roc_curve(y_test, prob_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=fpr, y=tpr, z=np.clip(thr, 0, 1), mode="lines",
        line=dict(color=ORANGE, width=7),
        name=f"Gradient Boosting (AUC = {roc_auc_score(y_test, prob_test):.3f})"))
    fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 1], z=[0.5, 0.5], mode="lines",
                               line=dict(color="gray", width=4, dash="dash"),
                               name="Random guess"))
    fig.update_layout(scene=dict(xaxis_title="False positive rate",
                                 yaxis_title="True positive rate",
                                 zaxis_title="Threshold"))
    show(fig, 600)

    st.subheader("3D confusion matrix")
    threshold = st.slider("Decision threshold", 0.05, 0.60, float(THRESHOLD), 0.005)
    pred = (prob_test >= threshold).astype(int)
    labels = ["Not readmitted <30", "Readmitted <30"]
    cm = confusion_matrix(y_test, pred)
    show(bars3d(labels, labels, cm, "Patients", text_fmt="{:,.0f}",
                xtitle="Predicted", ytitle="Actual"), 540)

    c1, c2, c3 = st.columns(3)
    c1.metric("Recall", f"{recall_score(y_test, pred, zero_division=0)*100:.1f}%")
    c2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0)*100:.1f}%")
    c3.metric("F1 score", f"{f1_score(y_test, pred, zero_division=0)*100:.1f}%")
    c4, c5, c6 = st.columns(3)
    c4.metric("Accuracy", f"{accuracy_score(y_test, pred)*100:.1f}%")
    c5.metric("ROC-AUC", f"{roc_auc_score(y_test, prob_test):.3f}")
    c6.metric("PR-AUC", f"{average_precision_score(y_test, prob_test):.3f}")
    st.caption(f"The notebook's chosen threshold is {THRESHOLD}. Moving the slider "
               "shows the recall vs precision trade-off.")

    st.subheader("Top feature importances")
    imp = bundle["importance"].head(10)
    show(bars3d(imp.index, [""], [imp.values * 100], "Importance (%)",
                text_fmt="{:.1f}", xtitle="Feature"), 520)

# ---------- PAGE: 3D EXPLORER ----------
def explorer3d():
    banner("3D Explorer", "Real patients and the model's risk landscape")
    tab1, tab2 = st.tabs(["Patient cloud", "Risk surface (model)"])

    with tab1:
        st.subheader("Patients by stay, medications and prior inpatient visits")
        sample = df.sample(min(3000, len(df)), random_state=1).copy()
        rng = np.random.default_rng(1)
        for c in ["time_in_hospital", "num_medications", "number_inpatient"]:
            sample[c] = sample[c] + rng.normal(0, 0.15, len(sample))
        fig = px.scatter_3d(sample, x="time_in_hospital", y="num_medications",
                            z="number_inpatient", color="readmitted",
                            color_discrete_sequence=PALETTE, opacity=0.75)
        fig.update_traces(marker=dict(size=3.5))
        fig.update_layout(scene=dict(xaxis_title="Days in hospital",
                                     yaxis_title="Number of medications",
                                     zaxis_title="Prior inpatient visits"))
        show(fig, 600)
        st.caption("3,000 random patients. A tiny random jitter spreads out overlapping "
                   "points. It is for display only.")

    with tab2:
        st.subheader("How the model's predicted risk changes")
        opts = ["time_in_hospital", "num_medications", "num_lab_procedures",
                "number_inpatient", "number_diagnoses"]
        c1, c2 = st.columns(2)
        xf = c1.selectbox("X axis", opts, index=0)
        yf = c2.selectbox("Y axis", opts, index=3)
        if xf == yf:
            st.warning("Pick two different variables.")
        else:
            xs = np.linspace(df[xf].quantile(0.01), df[xf].quantile(0.99), 25)
            ys = np.linspace(df[yf].quantile(0.01), df[yf].quantile(0.99), 25)
            gx, gy = np.meshgrid(xs, ys)
            grid = pd.DataFrame([DEFAULTS] * gx.size)[FEATURES]
            grid[xf] = gx.ravel()
            grid[yf] = gy.ravel()
            z = model.predict_proba(grid)[:, 1].reshape(gx.shape) * 100
            fig = go.Figure(go.Surface(
                x=xs, y=ys, z=z,
                colorscale=[[0, TEAL], [0.5, ORANGE], [1, RED]],
                colorbar=dict(title="Risk %")))
            fig.update_layout(scene=dict(xaxis_title=xf, yaxis_title=yf,
                                         zaxis_title="Predicted risk (%)"))
            show(fig, 600)
            st.caption("Every other input is held at the typical value in the training "
                       "data. These are predictions for hypothetical patients.")

# ---------- PATIENT HEALTHCARE DATA ----------
HEALTH_PATH = find_data_file("healthcare_dataset.csv")

@st.cache_data
def load_health():
    return load_health_clean(HEALTH_PATH)

def health_filters(h):
    """Sidebar filters. Returns the filtered frame."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    conds = sorted(h["Medical Condition"].unique())
    adms = sorted(h["Admission Type"].unique())
    genders = sorted(h["Gender"].unique())
    sel_c = st.sidebar.multiselect("Medical condition", conds, default=conds)
    sel_a = st.sidebar.multiselect("Admission type", adms, default=adms)
    sel_g = st.sidebar.multiselect("Gender", genders, default=genders)
    y0, y1 = int(h["Admission Year"].min()), int(h["Admission Year"].max())
    yr = st.sidebar.slider("Admission year", y0, y1, (y0, y1))
    return h[h["Medical Condition"].isin(sel_c) & h["Admission Type"].isin(sel_a)
             & h["Gender"].isin(sel_g) & h["Admission Year"].between(yr[0], yr[1])]

def avg_chart(h, by, col, label, fmt, xtitle):
    g = h.groupby(by, observed=True)[col].mean().sort_values(ascending=False)
    show(bars3d(g.index, [""], [g.values], label, text_fmt=fmt, xtitle=xtitle))
    return g

def grid_chart(table, label, fmt, xtitle, ytitle, height=560):
    show(bars3d(table.columns, table.index, table.values, label,
                text_fmt=fmt, xtitle=xtitle, ytitle=ytitle), height)

# ---------- PAGE: HOSPITAL OVERVIEW ----------
def hospital_overview(h):
    banner("Patient Healthcare Dashboard", "Hospital admissions overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Admissions", f"{len(h):,}")
    c2.metric("Avg length of stay", f"{h['Length of Stay'].mean():.1f} days")
    c3.metric("Avg billing", f"${h['Billing Amount'].mean():,.0f}")
    c4.metric("Top condition", h["Medical Condition"].value_counts().idxmax())
    c5.metric("Abnormal results", f"{(h['Test Results'] == 'Abnormal').mean()*100:.1f}%")

    st.subheader("Admissions by year")
    yearly = h.groupby("Admission Year").size()
    show(bars3d(yearly.index, [""], [yearly.values], "Admissions",
                text_fmt="{:,.0f}", xtitle="Year"))
    st.caption(f"Data covers {h['Date of Admission'].min():%d %b %Y} to "
               f"{h['Date of Admission'].max():%d %b %Y}, so the first and last "
               "years are partial.")

    st.subheader("Admissions by month")
    monthly = h.groupby("Admission Month").size().reindex(range(1, 13), fill_value=0)
    show(bars3d(list(calendar.month_abbr)[1:], [""], [monthly.values], "Admissions",
                text_fmt="{:,.0f}", xtitle="Month"))

# ---------- PAGE: CONDITIONS ----------
def conditions(h):
    banner("Medical Conditions", "Volume, stay, billing, tests and medication by condition")

    st.subheader("Admissions by medical condition")
    cnt = h["Medical Condition"].value_counts()
    show(bars3d(cnt.index, [""], [cnt.values], "Admissions",
                text_fmt="{:,.0f}", xtitle="Condition"))

    st.subheader("Average length of stay by condition")
    avg_chart(h, "Medical Condition", "Length of Stay", "Days", "{:.1f}", "Condition")

    st.subheader("Average billing by condition")
    avg_chart(h, "Medical Condition", "Billing Amount", "Billing ($)",
              "${:,.0f}", "Condition")

    summary = (h.groupby("Medical Condition")
               .agg(Admissions=("Name", "size"),
                    Avg_stay_days=("Length of Stay", "mean"),
                    Avg_billing=("Billing Amount", "mean"))
               .round(2).sort_values("Avg_billing", ascending=False))
    st.dataframe(summary)
    st.caption("The averages are close across conditions, so use the table to see "
               "the exact differences.")

    st.subheader("Test results by condition (% of each condition)")
    tr = pd.crosstab(h["Medical Condition"], h["Test Results"], normalize="index") * 100
    grid_chart(tr, "% of condition", "{:.0f}", "Test result", "Condition")

    st.subheader("Medication by condition (% of each condition)")
    med = pd.crosstab(h["Medical Condition"], h["Medication"], normalize="index") * 100
    grid_chart(med, "% of condition", "{:.0f}", "Medication", "Condition")

# ---------- PAGE: ADMISSIONS & INSURANCE ----------
def admissions_insurance(h):
    banner("Admissions and Insurance", "How admission type and insurer relate to stay and cost")

    st.subheader("Admissions by type")
    cnt = h["Admission Type"].value_counts()
    show(bars3d(cnt.index, [""], [cnt.values], "Admissions",
                text_fmt="{:,.0f}", xtitle="Admission type"))

    st.subheader("Average length of stay by admission type")
    g = avg_chart(h, "Admission Type", "Length of Stay", "Days", "{:.1f}", "Admission type")
    st.dataframe(g.round(2).rename("Avg stay (days)"))

    st.subheader("Average stay: condition by admission type")
    pv = h.pivot_table(index="Medical Condition", columns="Admission Type",
                       values="Length of Stay", aggfunc="mean")
    grid_chart(pv, "Days", "{:.1f}", "Admission type", "Condition")

    st.subheader("Average billing by insurance provider")
    g = avg_chart(h, "Insurance Provider", "Billing Amount", "Billing ($)",
                  "${:,.0f}", "Insurance provider")
    st.dataframe(g.round(2).rename("Avg billing ($)"))

    st.subheader("Average billing: condition by insurance provider")
    pv = h.pivot_table(index="Medical Condition", columns="Insurance Provider",
                       values="Billing Amount", aggfunc="mean")
    grid_chart(pv, "Billing ($)", "${:,.0f}", "Insurance provider", "Condition")

# ---------- PAGE: DEMOGRAPHICS ----------
def demographics(h):
    banner("Patient Demographics", "Age and gender patterns")

    st.subheader("Admissions by age group")
    cnt = h["Age Group"].value_counts().sort_index()
    show(bars3d(cnt.index, [""], [cnt.values], "Admissions",
                text_fmt="{:,.0f}", xtitle="Age group"))

    st.subheader("Average length of stay by age group")
    g = h.groupby("Age Group", observed=True)["Length of Stay"].mean()
    show(bars3d(g.index, [""], [g.values], "Days", text_fmt="{:.1f}",
                xtitle="Age group"))
    st.dataframe(g.round(2).rename("Avg stay (days)"))

    st.subheader("Admissions by gender")
    gc = h["Gender"].value_counts()
    show(bars3d(gc.index, [""], [gc.values], "Admissions",
                text_fmt="{:,.0f}", xtitle="Gender"))

    st.subheader("Condition by gender (admissions)")
    ct = pd.crosstab(h["Medical Condition"], h["Gender"])
    grid_chart(ct, "Admissions", "{:,.0f}", "Gender", "Condition", 520)

    st.subheader("Condition by age group (admissions)")
    ct = pd.crosstab(h["Medical Condition"], h["Age Group"])
    grid_chart(ct, "Admissions", "{:,.0f}", "Age group", "Condition")

# ---------- ROUTER (always keep at the bottom) ----------
st.sidebar.title("🏥 Navigation")
section = st.sidebar.radio("Section", ["Patient Healthcare", "Readmission Risk"])

if section == "Patient Healthcare":
    page = st.sidebar.radio("Page", ["Hospital Overview", "Conditions",
                                     "Admissions & Insurance", "Demographics"])
    if not os.path.exists(HEALTH_PATH):
        st.error("Missing file: healthcare_dataset.csv. Put it in the data folder "
                 "(or next to app.py), then rerun.")
    else:
        hdf = health_filters(load_health())
        if hdf.empty:
            st.warning("No records match the current filters.")
        elif page == "Hospital Overview":
            hospital_overview(hdf)
        elif page == "Conditions":
            conditions(hdf)
        elif page == "Admissions & Insurance":
            admissions_insurance(hdf)
        else:
            demographics(hdf)
else:
    page = st.sidebar.radio("Page", ["Overview", "EDA", "Risk Estimator",
                                     "Model Comparison", "Model Evaluation",
                                     "3D Explorer"])
    if page == "Overview":
        overview()
    elif page == "EDA":
        eda()
    elif page == "Risk Estimator":
        estimator()
    elif page == "Model Comparison":
        comparison()
    elif page == "Model Evaluation":
        evaluation()
    else:
        explorer3d()
