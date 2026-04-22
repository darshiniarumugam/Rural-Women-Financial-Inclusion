# ================================================================
# Financial Inclusion of Rural Women in India
# ABA Final Project  |  NFHS-5 (2019-21) + RBI (2021)
# Sections: Overview | Predict | Insights
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinInclude India",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 3rem 2.5rem; max-width: 1380px; }

/* ── Topbar ── */
.topbar {
    background: #0a1628;
    padding: 0.9rem 2.5rem;
    margin: -1rem -2.5rem 2.5rem -2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #1a2f4a;
}
.topbar-title { color: #e2e8f0; font-size: 1rem; font-weight: 600; letter-spacing: -0.01em; }
.topbar-sub   { color: #475569; font-size: 0.72rem; margin-top: 2px; }
.topbar-pill  {
    background: #1a2f4a;
    color: #38bdf8;
    font-size: 0.68rem;
    font-weight: 500;
    padding: 3px 12px;
    border-radius: 99px;
    border: 1px solid #1e4068;
    letter-spacing: 0.02em;
}

/* ── KPI card ── */
.kpi {
    background: #0a1628;
    border: 1px solid #1a2f4a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
}
.kpi-label { color: #475569; font-size: 0.68rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.45rem; }
.kpi-value { color: #f1f5f9; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; }
.kpi-sub   { color: #38bdf8; font-size: 0.7rem; margin-top: 0.35rem; }

/* ── Section heading ── */
.sh {
    color: #e2e8f0;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid #1a2f4a;
}

/* ── Info card ── */
.info {
    background: #0d1f35;
    border: 1px solid #1a2f4a;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.65;
    margin: 0.5rem 0;
}
.info strong { color: #e2e8f0; }

/* ── Prediction result box ── */
.result-box {
    background: #0d1f35;
    border: 1px solid #1a2f4a;
    border-radius: 12px;
    padding: 2rem 1.5rem;
    text-align: center;
}
.result-num  { color: #38bdf8; font-size: 3.2rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
.result-lbl  { color: #475569; font-size: 0.78rem; margin-top: 0.4rem; }
.result-diff { font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem; }

/* ── Upload area styling ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #1a2f4a !important;
    border-radius: 8px;
    background: #0a1628;
}

/* ── Table ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Nav radio ── */
div[role="radiogroup"] { gap: 0.5rem; }
div[role="radiogroup"] label {
    background: #0a1628 !important;
    border: 1px solid #1a2f4a !important;
    border-radius: 6px !important;
    padding: 0.35rem 1.1rem !important;
    color: #64748b !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    transition: all 0.15s;
}
div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label:has(input:checked) {
    background: #1a2f4a !important;
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Chart theme ─────────────────────────────────────────────
BG  = "#0a1628"
C1  = "#38bdf8"
C2  = "#6366f1"
C3  = "#22c55e"
C4  = "#ef4444"
AM  = "#f59e0b"
GR  = "#1a2f4a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GR, "axes.labelcolor": "#64748b",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "text.color": "#e2e8f0", "grid.color": GR,
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "sans-serif", "axes.titlecolor": "#e2e8f0",
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 8.5, "figure.dpi": 130,
})

# ── Embedded dataset ─────────────────────────────────────────
@st.cache_data
def load_base():
    return pd.DataFrame({
        "State": [
            "Andaman & Nicobar","Andhra Pradesh","Arunachal Pradesh","Assam","Bihar",
            "Chhattisgarh","Dadra & NH + DD","Goa","Gujarat","Haryana","Himachal Pradesh",
            "Jammu & Kashmir","Jharkhand","Karnataka","Kerala","Ladakh","Madhya Pradesh",
            "Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Puducherry",
            "Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
            "Uttar Pradesh","Uttarakhand","West Bengal",
        ],
        "BankAccount": [
            89.75,79.63,76.74,77.86,76.21,81.14,89.28,92.38,67.48,72.39,82.23,
            83.52,79.75,87.68,78.22,88.65,73.34,70.91,70.71,68.22,74.97,55.41,
            87.38,96.72,82.10,79.00,76.70,91.72,85.21,77.67,74.06,79.75,73.24,
        ],
        "BranchDensity": [
            0.1703,0.1345,0.1045,0.0830,0.0598,0.0959,0.1230,0.4529,0.1225,0.1770,
            0.2238,0.1296,0.0821,0.1595,0.1936,0.2345,0.0847,0.1112,0.0591,0.1057,
            0.1638,0.0819,0.1176,0.1576,0.2234,0.0978,0.2414,0.1548,0.1448,0.1355,
            0.0802,0.1918,0.0955,
        ],
        "HHDecision": [
            95.52,84.30,86.64,91.77,87.02,91.54,87.43,98.60,90.67,86.15,93.90,81.66,
            89.84,80.52,94.61,80.27,84.13,89.15,95.00,91.98,98.02,99.82,90.32,96.55,
            90.25,86.79,93.94,93.65,86.16,89.47,86.78,90.64,85.81,
        ],
        "ChildMarriage": [
            15.27,32.90,19.33,33.35,43.36,13.20,26.16,3.18,26.91,13.65,5.14,5.26,
            36.09,24.66,8.22,3.13,26.55,27.57,17.61,19.12,14.00,7.25,21.71,0.97,
            8.68,28.28,12.54,15.18,27.38,42.40,17.85,9.76,48.13,
        ],
        "Literacy": [
            85.58,63.84,71.63,75.39,54.54,71.30,67.90,93.43,69.03,78.75,91.23,74.71,
            59.34,70.95,97.47,76.60,63.30,79.51,84.84,85.53,87.67,82.74,69.22,89.07,
            80.04,61.63,86.24,81.50,58.10,76.88,65.55,79.78,72.54,
        ],
        "MobileOwn": [
            80.86,40.93,75.28,53.87,49.26,33.94,45.53,87.06,36.18,43.41,77.77,73.25,
            43.67,53.42,86.90,81.22,31.44,43.07,68.19,64.29,70.64,76.33,47.95,76.56,
            54.90,45.30,83.25,68.93,50.60,48.03,42.37,55.65,39.07,
        ],
        "InternetUse": [
            27.87,15.35,49.61,24.36,17.00,20.78,23.84,68.28,17.53,42.80,45.23,38.90,
            22.69,24.76,57.54,53.98,20.08,23.71,40.39,27.95,47.98,40.26,21.33,50.37,
            48.75,30.76,68.08,39.20,15.78,17.66,24.48,39.35,13.98,
        ],
        "PaidCash": [
            17.12,44.50,23.55,19.27,12.80,42.62,39.47,27.86,34.05,17.03,17.71,18.50,
            17.74,41.36,25.79,28.25,28.00,39.60,44.03,39.11,29.00,20.67,25.58,44.84,
            20.21,17.52,29.25,45.51,55.54,25.93,14.83,22.22,20.17,
        ],
        "AssetOwn": [
            19.36,50.63,70.25,43.85,55.67,45.45,59.74,24.07,43.27,41.03,23.38,60.77,
            66.47,69.65,29.18,72.95,41.27,24.50,58.93,70.11,28.36,28.86,45.47,38.30,
            67.07,26.61,50.57,52.04,74.46,17.28,53.53,25.13,22.46,
        ],
    })

df = load_base()

# ── Model training ───────────────────────────────────────────
FEATURES = ["BranchDensity", "HHDecision", "ChildMarriage"]

# OLS coefficients from regression analysis (Colab)
OLS = {
    "intercept":    110.411,
    "BranchDensity": 56.3041,
    "HHDecision":    -0.4251,
    "ChildMarriage": -0.0479,
}

@st.cache_resource
def train_models():
    X = df[FEATURES].values
    y = df["BankAccount"].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    rf.fit(Xs, y)
    gb.fit(Xs, y)
    ols_p = (OLS["intercept"]
             + OLS["BranchDensity"] * df["BranchDensity"]
             + OLS["HHDecision"]    * df["HHDecision"]
             + OLS["ChildMarriage"] * df["ChildMarriage"]).values
    ens = (ols_p + rf.predict(Xs) + gb.predict(Xs)) / 3
    return sc, rf, gb, ols_p, ens, y

sc, rf_model, gb_model, ols_pred, ens_pred, y_arr = train_models()

# ── Prediction helper ────────────────────────────────────────
def predict_row(bd, hh, cm):
    Xs = sc.transform([[bd, hh, cm]])
    ols_val = (OLS["intercept"]
               + OLS["BranchDensity"] * bd
               + OLS["HHDecision"]    * hh
               + OLS["ChildMarriage"] * cm)
    p = (ols_val + rf_model.predict(Xs)[0] + gb_model.predict(Xs)[0]) / 3
    return round(float(np.clip(p, 40, 100)), 2)

def predict_df(data):
    preds = []
    for _, row in data.iterrows():
        preds.append(predict_row(row["BranchDensity"], row["HHDecision"], row["ChildMarriage"]))
    return preds

# ── Column aliases ───────────────────────────────────────────
ALIASES = {
    "branchdensity": "BranchDensity",
    "branch_density": "BranchDensity",
    "branch density": "BranchDensity",
    "hhdecision": "HHDecision",
    "hh_decision": "HHDecision",
    "hh decision": "HHDecision",
    "household decision": "HHDecision",
    "householddecision": "HHDecision",
    "childmarriage": "ChildMarriage",
    "child_marriage": "ChildMarriage",
    "child marriage": "ChildMarriage",
}

def resolve_columns(uploaded_df):
    rename_map = {}
    for col in uploaded_df.columns:
        key = col.strip().lower()
        if key in ALIASES:
            rename_map[col] = ALIASES[key]
    return uploaded_df.rename(columns=rename_map)

# ── Topbar ────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div>
    <div class="topbar-title">FinInclude India</div>
    <div class="topbar-sub">Financial Inclusion of Rural Women &nbsp;|&nbsp; NFHS-5 (2019-21) + RBI (2021) &nbsp;|&nbsp; ABA Final Project</div>
  </div>
  <span class="topbar-pill">33 States / UTs</span>
</div>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────
nav = st.radio(
    label="",
    options=["Overview", "Predict", "Insights"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<hr style='border:none;border-top:1px solid #1a2f4a;margin:.25rem 0 2rem'>",
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════
if nav == "Overview":

    # State selector
    states = sorted(df["State"].tolist())
    sel = st.selectbox("Select State / UT", states, index=states.index("Tamil Nadu"))
    row = df[df["State"] == sel].iloc[0]
    avg = df.mean(numeric_only=True)
    rank = int(df["BankAccount"].rank(ascending=False)[df["State"] == sel].values[0])

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Bank Account Usage", f"{row['BankAccount']:.1f}%", f"Rank {rank} of 33"),
        (c2, "Branch Density",     f"{row['BranchDensity']:.4f}",
         f"{row['BranchDensity'] - avg['BranchDensity']:+.4f} vs average"),
        (c3, "HH Decision Making", f"{row['HHDecision']:.1f}%",
         f"{row['HHDecision'] - avg['HHDecision']:+.1f}% vs average"),
        (c4, "Child Marriage Rate", f"{row['ChildMarriage']:.1f}%",
         f"{row['ChildMarriage'] - avg['ChildMarriage']:+.1f}% vs average"),
    ]:
        with col:
            st.markdown(
                f'''<div class="kpi">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-sub">{sub}</div>
                </div>''',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.25, 1])

    with col_left:
        st.markdown(f'<div class="sh">{sel} — All Indicators vs National Average</div>',
                    unsafe_allow_html=True)

        display_vars = [
            ("BankAccount",   "Bank Account Usage",       False, False),
            ("BranchDensity", "Branch Density (per 1K)",  True,  False),
            ("HHDecision",    "HH Decision Making",       False, False),
            ("ChildMarriage", "Child Marriage Rate",       False, True),
            ("Literacy",      "Female Literacy",           False, False),
            ("MobileOwn",     "Mobile Ownership",          False, False),
            ("InternetUse",   "Internet Usage",            False, False),
            ("PaidCash",      "Women Paid in Cash",        False, False),
            ("AssetOwn",      "Asset Ownership",           False, False),
        ]

        for var, label, is_density, lower_better in display_vars:
            v    = row[var]
            navg = avg[var]
            diff = v - navg
            if lower_better:
                better = diff <= 0
            else:
                better = diff >= 0
            color  = C3 if better else C4
            max_v  = df[var].max()
            bar_w  = int((v / max_v) * 100) if max_v > 0 else 0
            fmt    = f"{v:.4f}" if is_density else f"{v:.1f}"
            unit   = "/1K" if is_density else "%"

            st.markdown(f"""
            <div style="margin:10px 0">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="color:#64748b;font-size:.78rem">{label}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:.76rem;color:{color}">
                  {fmt}{unit}&nbsp;&nbsp;({diff:+.2f})</span>
              </div>
              <div style="background:{GR};border-radius:3px;height:6px;overflow:hidden">
                <div style="width:{bar_w}%;background:{color};height:100%;border-radius:3px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="sh">All States — Bank Account Usage (%)</div>',
                    unsafe_allow_html=True)
        ranked = df.sort_values("BankAccount", ascending=True)
        fig, ax = plt.subplots(figsize=(5.5, 9.2))
        avg_val = avg["BankAccount"]
        bar_colors = [AM if s == sel else (C3 if v >= avg_val else C1)
                      for s, v in zip(ranked["State"], ranked["BankAccount"])]
        bars = ax.barh(ranked["State"], ranked["BankAccount"],
                       color=bar_colors, edgecolor="none", height=0.72)
        ax.axvline(avg_val, color=C2, linestyle="--", lw=1.4,
                   label=f"Average {avg_val:.1f}%")
        for bar, val in zip(bars, ranked["BankAccount"]):
            ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", fontsize=7, color="#64748b")
        ax.set_xlabel("Bank Account Usage (%)")
        ax.set_xlim(45, 108)
        ax.set_title(f"Highlighted: {sel}", fontsize=9)
        ax.legend(facecolor=BG, edgecolor=GR, labelcolor="#94a3b8", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # National summary row
    st.markdown('<div class="sh" style="margin-top:2rem">National Summary</div>',
                unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns(4)
    for col, label, value in [
        (n1, "National Average",   f"{avg['BankAccount']:.1f}%"),
        (n2, "Highest State",
         f"{df.loc[df['BankAccount'].idxmax(), 'State']} ({df['BankAccount'].max():.1f}%)"),
        (n3, "Lowest State",
         f"{df.loc[df['BankAccount'].idxmin(), 'State']} ({df['BankAccount'].min():.1f}%)"),
        (n4, "States Above Average",
         f"{(df['BankAccount'] > avg['BankAccount']).sum()} / 33"),
    ]:
        with col:
            st.markdown(
                f'''<div class="kpi">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value" style="font-size:1.15rem">{value}</div>
                </div>''',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════
# SECTION 2 — PREDICT
# ═══════════════════════════════════════════════════════════
elif nav == "Predict":

    st.markdown('<div class="sh">Predict Bank Account Usage</div>', unsafe_allow_html=True)

    tab_upload, tab_manual = st.tabs(["Upload Dataset", "Manual Input"])

    # ── Tab 1: Upload ────────────────────────────────────────
    with tab_upload:
        st.markdown("""
        <div class="info">
          Upload any <strong>CSV or Excel</strong> file containing the three predictor columns.
          The model will predict <strong>Bank Account Usage (%)</strong> for each row.<br><br>
          Required columns (case-insensitive, underscores accepted):
          <strong>BranchDensity</strong>, <strong>HHDecision</strong>, <strong>ChildMarriage</strong>
        </div>
        """, unsafe_allow_html=True)

        # Download template
        template_df = pd.DataFrame({
            "State":         ["Example State A", "Example State B"],
            "BranchDensity": [0.14, 0.08],
            "HHDecision":    [90.0, 85.5],
            "ChildMarriage": [15.0, 35.0],
        })
        csv_template = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Template CSV",
            data=csv_template,
            file_name="prediction_template.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="File must contain BranchDensity, HHDecision, and ChildMarriage columns.",
        )

        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    user_df = pd.read_csv(uploaded)
                else:
                    user_df = pd.read_excel(uploaded)

                user_df = resolve_columns(user_df)
                missing = [c for c in FEATURES if c not in user_df.columns]

                if missing:
                    st.error(f"Could not find required columns: {missing}. "
                             f"Please rename them to: BranchDensity, HHDecision, ChildMarriage")
                else:
                    for col in FEATURES:
                        user_df[col] = pd.to_numeric(user_df[col], errors="coerce")

                    n_bad = user_df[FEATURES].isnull().any(axis=1).sum()
                    if n_bad > 0:
                        st.warning(f"{n_bad} row(s) had non-numeric values and will be skipped.")

                    clean_df = user_df.dropna(subset=FEATURES).copy()

                    if len(clean_df) == 0:
                        st.error("No valid rows to predict after removing missing values.")
                    else:
                        clean_df["Predicted_BankAccount_%"] = predict_df(clean_df)
                        nat_avg = df["BankAccount"].mean()
                        clean_df["vs_National_Avg"] = (
                            clean_df["Predicted_BankAccount_%"] - nat_avg
                        ).round(2)
                        clean_df["Inclusion_Zone"] = clean_df["Predicted_BankAccount_%"].apply(
                            lambda x: "High (>85%)" if x > 85 else
                                      "Good (75-85%)" if x > 75 else
                                      "Moderate (65-75%)" if x > 65 else "Low (<65%)"
                        )

                        st.markdown('<div class="sh" style="margin-top:1.5rem">Prediction Results</div>',
                                    unsafe_allow_html=True)

                        col_s, col_a, col_b, col_c = st.columns(4)
                        with col_s:
                            st.markdown(f'''<div class="kpi"><div class="kpi-label">Rows Predicted</div>
                              <div class="kpi-value">{len(clean_df)}</div></div>''',
                              unsafe_allow_html=True)
                        with col_a:
                            st.markdown(f'''<div class="kpi"><div class="kpi-label">Average Predicted</div>
                              <div class="kpi-value">{clean_df["Predicted_BankAccount_%"].mean():.1f}%</div></div>''',
                              unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f'''<div class="kpi"><div class="kpi-label">Highest Predicted</div>
                              <div class="kpi-value">{clean_df["Predicted_BankAccount_%"].max():.1f}%</div></div>''',
                              unsafe_allow_html=True)
                        with col_c:
                            st.markdown(f'''<div class="kpi"><div class="kpi-label">Lowest Predicted</div>
                              <div class="kpi-value">{clean_df["Predicted_BankAccount_%"].min():.1f}%</div></div>''',
                              unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        display_cols = (
                            ["State"] if "State" in clean_df.columns else []
                        ) + FEATURES + ["Predicted_BankAccount_%", "vs_National_Avg", "Inclusion_Zone"]
                        st.dataframe(
                            clean_df[display_cols].style.format({
                                "BranchDensity": "{:.4f}",
                                "HHDecision":    "{:.1f}",
                                "ChildMarriage": "{:.1f}",
                                "Predicted_BankAccount_%": "{:.2f}",
                                "vs_National_Avg": "{:+.2f}",
                            }).background_gradient(
                                subset=["Predicted_BankAccount_%"],
                                cmap="Blues",
                            ),
                            use_container_width=True,
                        )

                        # Download results
                        result_csv = clean_df[display_cols].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Results as CSV",
                            data=result_csv,
                            file_name="prediction_results.csv",
                            mime="text/csv",
                        )

                        # Chart if <=30 rows
                        if len(clean_df) <= 30 and "State" in clean_df.columns:
                            st.markdown('<div class="sh" style="margin-top:1.5rem">Prediction Chart</div>',
                                        unsafe_allow_html=True)
                            plot_df = clean_df.sort_values("Predicted_BankAccount_%", ascending=True)
                            fig2, ax2 = plt.subplots(figsize=(8, max(3, len(plot_df) * 0.45)))
                            bar_c2 = [C3 if v > nat_avg else C1
                                      for v in plot_df["Predicted_BankAccount_%"]]
                            ax2.barh(plot_df["State"].astype(str), plot_df["Predicted_BankAccount_%"],
                                     color=bar_c2, edgecolor="none", height=0.65)
                            ax2.axvline(nat_avg, color=AM, linestyle="--", lw=1.4,
                                        label=f"National avg {nat_avg:.1f}%")
                            for i, v in enumerate(plot_df["Predicted_BankAccount_%"]):
                                ax2.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8, color="#64748b")
                            ax2.set_xlabel("Predicted Bank Account Usage (%)")
                            ax2.set_title("Predicted Values from Uploaded Data")
                            ax2.legend(facecolor=BG, edgecolor=GR, labelcolor="#94a3b8", fontsize=8)
                            ax2.grid(axis="x", alpha=0.3)
                            fig2.tight_layout()
                            st.pyplot(fig2)
                            plt.close()

            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ── Tab 2: Manual ────────────────────────────────────────
    with tab_manual:
        st.markdown("""
        <div class="info">
          Adjust the three predictors to simulate any state or scenario.
          Select a state to auto-fill from the training data.
        </div>
        """, unsafe_allow_html=True)

        col_form, col_result = st.columns([1.1, 1])

        with col_form:
            st.markdown("<br>", unsafe_allow_html=True)
            state_opts = ["-- Manual --"] + sorted(df["State"].tolist())
            chosen = st.selectbox("Auto-fill from state", state_opts)

            if chosen != "-- Manual --":
                r = df[df["State"] == chosen].iloc[0]
                def_bd = float(r["BranchDensity"])
                def_hh = float(r["HHDecision"])
                def_cm = float(r["ChildMarriage"])
            else:
                def_bd = float(df["BranchDensity"].mean())
                def_hh = float(df["HHDecision"].mean())
                def_cm = float(df["ChildMarriage"].mean())

            st.markdown("<br>", unsafe_allow_html=True)

            bd = st.slider(
                "Branch Density  (bank offices per 1,000 people)",
                min_value=0.04, max_value=0.50,
                value=def_bd, step=0.005, format="%.3f",
            )
            hh = st.slider(
                "Household Decision Making  (%)",
                min_value=75.0, max_value=100.0,
                value=def_hh, step=0.5,
            )
            cm = st.slider(
                "Child Marriage Rate  (%)  — lower is better",
                min_value=0.0, max_value=55.0,
                value=def_cm, step=0.5,
            )

            st.markdown("<br>**Preset scenarios**", unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            if p1.button("National Average"):
                bd = float(df["BranchDensity"].mean())
                hh = float(df["HHDecision"].mean())
                cm = float(df["ChildMarriage"].mean())
            if p2.button("Best — Goa"):
                bd, hh, cm = 0.4529, 98.60, 3.18
            if p3.button("Low Inclusion — Bihar"):
                bd, hh, cm = 0.0598, 87.02, 43.36

        with col_result:
            pred = predict_row(bd, hh, cm)
            nat  = float(df["BankAccount"].mean())
            diff = pred - nat
            diff_color = C3 if diff >= 0 else C4
            diff_sign  = "+" if diff >= 0 else ""
            zone = ("High (>85%)" if pred > 85 else
                    "Good (75-85%)" if pred > 75 else
                    "Moderate (65-75%)" if pred > 65 else "Low (<65%)")

            st.markdown(f"""
            <div class="result-box">
              <div class="kpi-label">Predicted Bank Account Usage</div>
              <div class="result-num">{pred}%</div>
              <div class="result-diff" style="color:{diff_color}">
                {diff_sign}{diff:.1f}% vs national average ({nat:.1f}%)
              </div>
              <div class="result-lbl" style="margin-top:.5rem">Zone: {zone}</div>
              <div class="result-lbl" style="margin-top:.3rem">
                Ensemble &mdash; OLS + Random Forest + Gradient Boosting
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Variable contribution breakdown
            st.markdown("<br>**OLS contribution by variable**", unsafe_allow_html=True)
            contribs = {
                "Branch Density":    OLS["BranchDensity"] * bd,
                "HH Decision Making": OLS["HHDecision"]   * hh,
                "Child Marriage":     OLS["ChildMarriage"] * cm,
            }
            for label, contrib in sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True):
                cc = C3 if contrib > 0 else C4
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            padding:5px 0;border-bottom:1px solid {GR}">
                  <span style="color:#64748b;font-size:.8rem">{label}</span>
                  <span style="font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:{cc}">
                    {contrib:+.2f}</span>
                </div>""", unsafe_allow_html=True)

            # Zone gauge
            st.markdown("<br>**Inclusion zone**", unsafe_allow_html=True)
            fig_g, ax_g = plt.subplots(figsize=(5, 1.4))
            zones = [(55, 65, "Low", C4), (65, 75, "Moderate", AM),
                     (75, 85, "Good", C1), (85, 100, "High", C3)]
            for s, e, z_lbl, z_col in zones:
                ax_g.barh(0, e - s, left=s, color=z_col, height=0.5, edgecolor="none", alpha=0.8)
                ax_g.text((s + e) / 2, 0, z_lbl, ha="center", va="center",
                          fontsize=7, color="white", fontweight="bold")
            ax_g.axvline(pred, color="white", lw=2.5, zorder=5)
            ax_g.text(pred, 0.31, f"{pred}%", ha="center", fontsize=9,
                      color="white", fontweight="bold")
            ax_g.set_xlim(50, 102)
            ax_g.set_ylim(-0.4, 0.6)
            ax_g.axis("off")
            fig_g.tight_layout()
            st.pyplot(fig_g)
            plt.close()


# ═══════════════════════════════════════════════════════════
# SECTION 3 — INSIGHTS
# ═══════════════════════════════════════════════════════════
elif nav == "Insights":

    st.markdown('<div class="sh">Data Insights and Model Performance</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1.05])

    with col_a:
        # Key drivers
        st.markdown('<div class="sh">Key Drivers of Financial Inclusion</div>', unsafe_allow_html=True)
        drivers = [
            ("Branch Density",
             "The strongest predictor. States with more bank offices per 1,000 people "
             "consistently show higher rural women's account usage. Physical banking "
             "infrastructure remains the primary supply-side barrier."),
            ("Household Decision Making",
             "Women who participate in household financial decisions are more likely to "
             "hold and actively use formal bank accounts. Autonomy and inclusion are "
             "directly linked."),
            ("Child Marriage Rate",
             "Higher child marriage rates correlate with lower financial inclusion. "
             "Early marriage interrupts girls' education and economic participation, "
             "reducing both the need and ability to use formal banking."),
        ]
        for title, text in drivers:
            st.markdown(f"""
            <div style="background:#0d1f35;border:1px solid {GR};border-radius:8px;
                        padding:.9rem 1.1rem;margin-bottom:.6rem">
              <div style="color:{C1};font-weight:600;font-size:.85rem;margin-bottom:5px">{title}</div>
              <div style="color:#64748b;font-size:.79rem;line-height:1.6">{text}</div>
            </div>""", unsafe_allow_html=True)

        # Model performance
        st.markdown('<div class="sh" style="margin-top:1.2rem">Model Performance</div>',
                    unsafe_allow_html=True)
        ols_r2 = r2_score(y_arr, ols_pred)
        rf_r2  = r2_score(y_arr, rf_model.predict(sc.transform(df[FEATURES].values)))
        ens_r2 = r2_score(y_arr, ens_pred)

        for name, r2_val, note in [
            ("OLS — 3 Variables",           ols_r2, "Linear model, interpretable"),
            ("Random Forest",                rf_r2,  "Non-linear, ensemble trees"),
            ("Ensemble (OLS + RF + GBM)",    ens_r2, "Combined — used for prediction"),
        ]:
            is_best = "Ensemble" in name
            bg = "#0d1f35" if is_best else "#0a1628"
            bc = f"{C1}55" if is_best else GR
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {bc};border-radius:8px;
                        padding:8px 14px;margin-bottom:5px;
                        display:flex;align-items:center;gap:10px">
              <span style="color:#e2e8f0;font-size:.81rem;flex:1">{name}</span>
              <span style="font-family:'IBM Plex Mono',monospace;color:{C1};
                           font-weight:600;font-size:.85rem">R&sup2; {r2_val:.3f}</span>
              <span style="color:#475569;font-size:.7rem;min-width:165px;text-align:right">{note}</span>
            </div>""", unsafe_allow_html=True)

        # Feature importance
        st.markdown('<div class="sh" style="margin-top:1.2rem">Variable Importance (Random Forest)</div>',
                    unsafe_allow_html=True)
        feat_imp = sorted(
            zip(FEATURES, rf_model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        for feat, score in feat_imp:
            bw = int(score * 600)
            labels = {
                "BranchDensity": "Branch Density",
                "HHDecision":    "HH Decision Making",
                "ChildMarriage": "Child Marriage Rate",
            }
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:8px 0">
              <span style="color:#64748b;font-size:.78rem;min-width:170px">{labels[feat]}</span>
              <div style="flex:1;background:{GR};border-radius:3px;height:7px;overflow:hidden">
                <div style="width:{min(bw,100)}%;background:{C1};height:100%;border-radius:3px"></div>
              </div>
              <span style="font-family:'IBM Plex Mono',monospace;font-size:.76rem;color:{C1};
                           min-width:38px;text-align:right">{score:.3f}</span>
            </div>""", unsafe_allow_html=True)

    with col_b:
        # Scatter: Branch Density vs Bank Account
        st.markdown('<div class="sh">Branch Density vs Bank Account Usage</div>',
                    unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5.8, 4.2))
        ax1.scatter(df["BranchDensity"], df["BankAccount"],
                    c=df["BankAccount"], cmap="Blues", vmin=50, vmax=100,
                    s=80, edgecolors=GR, lw=0.8, zorder=3)
        m1, b1 = np.polyfit(df["BranchDensity"], df["BankAccount"], 1)
        xl1 = np.linspace(df["BranchDensity"].min(), df["BranchDensity"].max(), 100)
        ax1.plot(xl1, m1 * xl1 + b1, color=AM, lw=1.8, linestyle="--", label="Trend")
        for _, r in df.iterrows():
            if r["BankAccount"] > 91 or r["BankAccount"] < 60 or r["BranchDensity"] > 0.35:
                ax1.annotate(r["State"][:9], (r["BranchDensity"], r["BankAccount"]),
                             xytext=(4, 3), textcoords="offset points",
                             fontsize=6, color="#475569")
        corr1 = df["BranchDensity"].corr(df["BankAccount"])
        ax1.set_xlabel("Branch Density (per 1,000 people)")
        ax1.set_ylabel("Bank Account Usage (%)")
        ax1.set_title(f"Pearson r = {corr1:.2f}")
        ax1.legend(facecolor=BG, edgecolor=GR, labelcolor="#94a3b8", fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close()

        # Scatter: Child Marriage vs Bank Account
        st.markdown('<div class="sh">Child Marriage Rate vs Bank Account Usage</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5.8, 4.2))
        ax2.scatter(df["ChildMarriage"], df["BankAccount"],
                    c=df["BankAccount"], cmap="Reds_r", vmin=50, vmax=100,
                    s=80, edgecolors=GR, lw=0.8, zorder=3)
        m2, b2 = np.polyfit(df["ChildMarriage"], df["BankAccount"], 1)
        xl2 = np.linspace(df["ChildMarriage"].min(), df["ChildMarriage"].max(), 100)
        ax2.plot(xl2, m2 * xl2 + b2, color=AM, lw=1.8, linestyle="--", label="Trend")
        for _, r in df.iterrows():
            if r["BankAccount"] > 91 or r["ChildMarriage"] > 40:
                ax2.annotate(r["State"][:9], (r["ChildMarriage"], r["BankAccount"]),
                             xytext=(4, 3), textcoords="offset points",
                             fontsize=6, color="#475569")
        corr2 = df["ChildMarriage"].corr(df["BankAccount"])
        ax2.set_xlabel("Child Marriage Rate (%)")
        ax2.set_ylabel("Bank Account Usage (%)")
        ax2.set_title(f"Pearson r = {corr2:.2f}")
        ax2.legend(facecolor=BG, edgecolor=GR, labelcolor="#94a3b8", fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Actual vs Predicted
        st.markdown('<div class="sh">Actual vs Predicted — Ensemble Model</div>',
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(5.8, 4.2))
        ax3.scatter(y_arr, ens_pred, color=C1, s=72,
                    edgecolors=GR, lw=0.8, alpha=0.92, zorder=3)
        mn = min(y_arr.min(), ens_pred.min()) - 2
        mx = max(y_arr.max(), ens_pred.max()) + 2
        ax3.plot([mn, mx], [mn, mx], color=AM, linestyle="--", lw=1.5, label="Perfect fit")
        ax3.set_xlabel("Actual (%)")
        ax3.set_ylabel("Predicted (%)")
        ax3.set_title(f"Ensemble  R\u00b2 = {ens_r2:.3f}")
        ax3.legend(facecolor=BG, edgecolor=GR, labelcolor="#94a3b8", fontsize=8)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # Policy recommendations
    st.markdown('<div class="sh" style="margin-top:2rem">Policy Recommendations</div>',
                unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    for col, title, text in [
        (r1, "Expand Banking Infrastructure",
         "Prioritise rural branch and Business Correspondent (BC) network expansion in "
         "low density states: Bihar (0.060), Uttar Pradesh (0.080), Jharkhand (0.082), "
         "Nagaland (0.082). Even modest improvements in branch density yield measurable "
         "gains in account usage."),
        (r2, "Strengthen Women's Financial Autonomy",
         "Embed financial literacy and account-opening drives within Self-Help Group "
         "programmes and Panchayat structures. Women who participate in household "
         "financial decisions are significantly more likely to use formal accounts."),
        (r3, "Reduce Child Marriage Through Education Incentives",
         "Target conditional cash transfer schemes in high child-marriage states "
         "(West Bengal 48%, Tripura 42%, Bihar 43%) to keep girls enrolled in school. "
         "Delayed marriage correlates directly with higher financial participation."),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0d1f35;border:1px solid {GR};border-radius:10px;
                        padding:1.1rem 1.2rem;height:100%">
              <div style="color:#e2e8f0;font-weight:600;font-size:.86rem;margin-bottom:.55rem">{title}</div>
              <div style="color:#64748b;font-size:.78rem;line-height:1.65">{text}</div>
            </div>""", unsafe_allow_html=True)
