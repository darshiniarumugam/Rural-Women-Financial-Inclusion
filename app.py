# ============================================================
# Financial Inclusion of Rural Women in India
# Professional Dashboard — Dataset Upload + Prediction
# NFHS-5 (2019-21) | RBI Branch Data (2021)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Inclusion — Rural Women India",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── STYLES ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 1400px; }

.topbar {
    background: #0f1923;
    padding: 1.1rem 2rem;
    margin: -1rem -2rem 2rem -2rem;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #1e3a52;
}
.topbar-left h1 { color: #e2e8f0; font-size: 1.05rem; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
.topbar-left p  { color: #64748b; font-size: 0.72rem; margin: 2px 0 0 0; }
.topbar-right   { display: flex; gap: 8px; align-items: center; }
.badge {
    background: #1e3a52; color: #38bdf8; font-size: 0.68rem;
    padding: 3px 10px; border-radius: 99px; font-weight: 500;
    border: 1px solid #38bdf822; letter-spacing: 0.03em;
}
.badge-green { background: #052e16; color: #4ade80; border-color: #16653422; }

.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: #0f1923; border: 1px solid #1e3a52;
    border-radius: 10px; padding: 1.1rem 1.3rem;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.kpi-label { color: #64748b; font-size: 0.68rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.4rem; }
.kpi-value { color: #f1f5f9; font-size: 1.9rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1; }
.kpi-sub   { color: #38bdf8; font-size: 0.72rem; margin-top: 0.3rem; }

.sec-head {
    color: #f1f5f9; font-size: 0.9rem; font-weight: 600;
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b; letter-spacing: -0.01em;
}

.info-box {
    background: #0d2137; border: 1px solid #1e3a52;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem; color: #94a3b8;
    font-size: 0.83rem; line-height: 1.65; margin: 0.5rem 0;
}
.info-box strong { color: #e2e8f0; }

.warn-box {
    background: #1c1400; border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem; color: #d97706;
    font-size: 0.83rem; line-height: 1.6; margin: 0.35rem 0;
}

.ok-box {
    background: #052e16; border-left: 3px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem; color: #4ade80;
    font-size: 0.83rem; line-height: 1.6; margin: 0.35rem 0;
}

.pred-card {
    background: #0d2137; border: 1px solid #38bdf833;
    border-radius: 14px; padding: 1.75rem 2rem;
    text-align: center; margin: 1.2rem 0;
}
.pred-main  { color: #38bdf8; font-size: 3.2rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
.pred-label { color: #64748b; font-size: 0.8rem; margin-top: 0.4rem; }
.pred-delta { font-size: 0.85rem; margin-top: 0.3rem; }

.model-row {
    display: flex; align-items: center; gap: 12px;
    padding: 0.65rem 1rem; border-radius: 7px;
    margin-bottom: 5px; background: #0f1923; border: 1px solid #1e3a52;
}
.model-best { background: #0d2137; border-color: #38bdf833; }
.model-name { color: #e2e8f0; font-weight: 500; font-size: 0.82rem; flex: 1; }
.model-val  { color: #38bdf8; font-weight: 600; font-size: 0.88rem; font-family: 'DM Mono', monospace; }

.upload-zone {
    border: 1.5px dashed #1e3a52; border-radius: 12px;
    padding: 2rem; text-align: center; background: #0f1923;
    margin: 1rem 0;
}
.upload-title { color: #e2e8f0; font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem; }
.upload-sub   { color: #64748b; font-size: 0.8rem; }

.req-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin: 1rem 0; }
.req-table th { background: #1e3a52; color: #94a3b8; padding: 7px 12px; text-align: left; font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
.req-table td { padding: 7px 12px; border-bottom: 1px solid #1e293b; color: #94a3b8; }
.req-table td:first-child { color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 0.78rem; }
.req-table tr:last-child td { border-bottom: none; }

.result-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 1rem; border-radius: 7px; margin-bottom: 4px;
    background: #0f1923; border: 1px solid #1e3a52;
}
.result-state { color: #e2e8f0; font-size: 0.83rem; font-weight: 500; }
.result-val   { color: #38bdf8; font-family: 'DM Mono', monospace; font-size: 0.85rem; font-weight: 600; }
.result-delta-pos { color: #4ade80; font-size: 0.75rem; }
.result-delta-neg { color: #f87171; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── COLOURS & CHART THEME ─────────────────────────────────────────────────
DARK    = '#0f1923'
ACCENT  = '#38bdf8'
ACCENT2 = '#818cf8'
ACCENT3 = '#4ade80'
DANGER  = '#f87171'
AMBER   = '#f59e0b'

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': DARK,
    'axes.edgecolor': '#1e3a52', 'axes.labelcolor': '#94a3b8',
    'xtick.color': '#64748b', 'ytick.color': '#64748b',
    'text.color': '#e2e8f0', 'grid.color': '#1e3a52',
    'grid.linestyle': '--', 'grid.alpha': 0.4,
    'font.family': 'sans-serif', 'axes.titlecolor': '#f1f5f9',
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'figure.dpi': 110,
})

# ── CONSTANTS ─────────────────────────────────────────────────────────────
FEATURES = ['BranchDensity', 'ChildMarriage', 'MobileOwn', 'PaidCash']
LABELS = {
    'BranchDensity': 'Branch Density (per 1,000 people)',
    'ChildMarriage':  'Child Marriage Rate (%)',
    'MobileOwn':      'Mobile Phone Ownership (%)',
    'PaidCash':       'Women Paid in Cash (%)',
}
TARGET = 'BankAccount'

REQUIRED_COLS = {
    'BankAccount':   'Women with bank account that they use (%) — DEPENDENT',
    'BranchDensity': 'Bank offices per 1,000 people',
    'ChildMarriage':  'Women married before age 18 (%)',
    'MobileOwn':      'Women with mobile phone they use (%)',
    'PaidCash':       'Women worked and paid in cash (%)',
}

# ── BUILT-IN REFERENCE DATA ───────────────────────────────────────────────
@st.cache_data
def get_reference_data():
    return pd.DataFrame({
        'State': [
            'Andaman & Nicobar', 'Andhra Pradesh', 'Arunachal Pradesh',
            'Assam', 'Bihar', 'Chhattisgarh', 'Dadra & NH + DD',
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
            'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Ladakh',
            'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
            'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab',
            'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
            'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
        ],
        'BankAccount':   [89.75,79.63,76.74,77.86,76.21,81.14,89.28,92.38,67.48,72.39,82.23,
                          83.52,79.75,87.68,78.22,88.65,73.34,70.91,70.71,68.22,74.97,55.41,
                          87.38,96.72,82.10,79.00,76.70,91.72,85.21,77.67,74.06,79.75,73.24],
        'BranchDensity': [0.1703,0.1345,0.1045,0.0830,0.0598,0.0959,0.1230,0.4529,0.1225,
                          0.1770,0.2238,0.1296,0.0821,0.1595,0.1936,0.2345,0.0847,0.1112,
                          0.0591,0.1057,0.1638,0.0819,0.1176,0.1576,0.2234,0.0978,0.2414,
                          0.1548,0.1448,0.1355,0.0802,0.1918,0.0955],
        'ChildMarriage': [15.27,32.90,19.33,33.35,43.36,13.20,26.16,3.18,26.91,13.65,5.14,
                          5.26,36.09,24.66,8.22,3.13,26.55,27.57,17.61,19.12,14.00,7.25,
                          21.71,0.97,8.68,28.28,12.54,15.18,27.38,42.40,17.85,9.76,48.13],
        'MobileOwn':     [80.86,40.93,75.28,53.87,49.26,33.94,45.53,87.06,36.18,43.41,77.77,
                          73.25,43.67,53.42,86.90,81.22,31.44,43.07,68.19,64.29,70.64,76.33,
                          47.95,76.56,54.90,45.30,83.25,68.93,50.60,48.03,42.37,55.65,39.07],
        'PaidCash':      [17.12,44.50,23.55,19.27,12.80,42.62,39.47,27.86,34.05,17.03,17.71,
                          18.50,17.74,41.36,25.79,28.25,28.00,39.60,44.03,39.11,29.00,20.67,
                          25.58,44.84,20.21,17.52,29.25,45.51,55.54,25.93,14.83,22.22,20.17],
    })

# ── MODEL TRAINING ────────────────────────────────────────────────────────
@st.cache_resource
def train_models(data_hash):
    df = get_reference_data()
    X  = df[FEATURES].values
    y  = df[TARGET].values

    # Leave-One-Out cross validation (best for small n)
    loo = LeaveOneOut()
    p_ols, p_rf, p_rd, p_gb = [], [], [], []
    for tr, te in loo.split(X):
        Xtr_c = sm.add_constant(X[tr], has_constant='add')
        Xte_c = sm.add_constant(X[te], has_constant='add')

        m_ols = sm.OLS(y[tr], Xtr_c).fit()
        p_ols.append(float(m_ols.predict(Xte_c)[0]))

        m_rf = RandomForestRegressor(n_estimators=200, random_state=42)
        m_rf.fit(X[tr], y[tr])
        p_rf.append(float(m_rf.predict(X[te])[0]))

        sc_ = StandardScaler()
        m_rd = Ridge(alpha=1.0)
        m_rd.fit(sc_.fit_transform(X[tr]), y[tr])
        p_rd.append(float(m_rd.predict(sc_.transform(X[te]))[0]))

        m_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        m_gb.fit(X[tr], y[tr])
        p_gb.append(float(m_gb.predict(X[te])[0]))

    p_ens = np.clip(np.mean([p_ols, p_rf, p_rd, p_gb], axis=0), 0, 100)

    def met(p): return r2_score(y, p), np.sqrt(mean_squared_error(y, p))
    r1,e1 = met(p_ols); r2,e2 = met(p_rf)
    r3,e3 = met(p_rd);  r4,e4 = met(p_gb); re,ee = met(p_ens)

    # Full models for prediction
    Xdf  = pd.DataFrame(X, columns=FEATURES)
    ys   = pd.Series(y)
    ols_f = sm.OLS(ys, sm.add_constant(Xdf)).fit()
    rf_f  = RandomForestRegressor(n_estimators=300, random_state=42); rf_f.fit(X, y)
    sc_f  = StandardScaler()
    rd_f  = Ridge(alpha=1.0); rd_f.fit(sc_f.fit_transform(X), y)
    gb_f  = GradientBoostingRegressor(n_estimators=200, random_state=42); gb_f.fit(X, y)
    imp   = pd.Series(rf_f.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return dict(
        r1=r1, e1=e1, r2=r2, e2=e2, r3=r3, e3=e3, r4=r4, e4=e4, re=re, ee=ee,
        ols=ols_f, rf=rf_f, sc=sc_f, rd=rd_f, gb=gb_f,
        imp=imp, y=y, p_ols=p_ols, p_rf=p_rf, p_rd=p_rd, p_gb=p_gb, p_ens=p_ens,
    )

R = train_models("v1")

def predict_row(row_array):
    inp    = np.array(row_array).reshape(1, -1)
    inp_df = pd.DataFrame(inp, columns=FEATURES)
    p1 = float(R['ols'].predict(sm.add_constant(inp_df, has_constant='add'))[0])
    p2 = float(R['rf'].predict(inp)[0])
    p3 = float(R['rd'].predict(R['sc'].transform(inp))[0])
    p4 = float(R['gb'].predict(inp)[0])
    return p1, p2, p3, p4, float(np.clip(np.mean([p1,p2,p3,p4]), 0, 100))

# ── TOP BAR ───────────────────────────────────────────────────────────────
ref_df = get_reference_data()
nat_avg = ref_df[TARGET].mean()

st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <h1>Financial Inclusion of Rural Women in India</h1>
    <p>Predictive Analytics Dashboard &nbsp;|&nbsp; NFHS-5 (2019-21) &nbsp;|&nbsp; RBI Branch Data (2021) &nbsp;|&nbsp; n = 33 States/UTs</p>
  </div>
  <div class="topbar-right">
    <span class="badge">ABA Final Project</span>
    <span class="badge-green badge">Model Ready</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── NAVIGATION ────────────────────────────────────────────────────────────
PAGES = ["Dashboard", "Upload Data & Predict", "Manual Predict", "Analysis", "Model Performance"]
page  = st.radio("", PAGES, horizontal=True, label_visibility="collapsed")
st.markdown("<hr style='border:none;border-top:1px solid #1e293b;margin:0.5rem 0 1.75rem'>",
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">National Average</div>
        <div class="kpi-value">{nat_avg:.1f}%</div>
        <div class="kpi-sub">Rural women bank account usage</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Highest State</div>
        <div class="kpi-value">{ref_df[TARGET].max():.1f}%</div>
        <div class="kpi-sub">Puducherry — leading inclusion</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Lowest State</div>
        <div class="kpi-value">{ref_df[TARGET].min():.1f}%</div>
        <div class="kpi-sub">Nagaland — needs intervention</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Model Accuracy (LOO CV)</div>
        <div class="kpi-value">{R['re']*100:.1f}%</div>
        <div class="kpi-sub">Ensemble R² x 100</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="sec-head">State-wise Bank Account Usage — Rural Women (%)</div>',
                    unsafe_allow_html=True)
        sdf = ref_df.sort_values(TARGET, ascending=True)
        fig, ax = plt.subplots(figsize=(8, 9))
        bar_colors = [ACCENT3 if v >= 85 else ACCENT if v >= 75 else DANGER
                      for v in sdf[TARGET]]
        ax.barh(sdf['State'], sdf[TARGET], color=bar_colors, edgecolor='none', height=0.7)
        ax.axvline(nat_avg, color=AMBER, linestyle='--', linewidth=1.2,
                   label=f'National avg: {nat_avg:.1f}%')
        for i, (val, _) in enumerate(zip(sdf[TARGET], sdf['State'])):
            ax.text(val + 0.3, i, f'{val:.1f}', va='center', fontsize=7, color='#94a3b8')
        ax.set_xlim(45, 105)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.set_xlabel('Bank Account Usage (%)')
        ax.legend(facecolor=DARK, edgecolor='#1e3a52', labelcolor='#94a3b8', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=ACCENT3, label='High >= 85%'),
                   mpatches.Patch(color=ACCENT,  label='Medium 75-85%'),
                   mpatches.Patch(color=DANGER,  label='Low < 75%'),
                   plt.Line2D([0],[0], color=AMBER, linestyle='--', linewidth=1.2,
                              label=f'Avg {nat_avg:.1f}%')]
        ax.legend(handles=patches, fontsize=7.5, loc='lower right',
                  facecolor=DARK, edgecolor='#1e3a52', labelcolor='#94a3b8')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="sec-head">Key Predictors (Feature Importance)</div>',
                    unsafe_allow_html=True)
        imp = R['imp']
        for feat, score in imp.items():
            bar_w = int(score * 500)
            st.markdown(f"""
            <div style="margin:8px 0">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="color:#94a3b8;font-size:.78rem">{LABELS[feat]}</span>
                <span style="font-family:'DM Mono',monospace;font-size:.78rem;color:{ACCENT}">{score:.3f}</span>
              </div>
              <div style="background:#1e3a52;border-radius:4px;height:7px;overflow:hidden">
                <div style="width:{min(bar_w,100)}%;background:{ACCENT};height:100%;border-radius:4px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head" style="margin-top:1.5rem">Research Hypotheses</div>',
                    unsafe_allow_html=True)
        pvals = dict(zip(FEATURES, R['ols'].pvalues[1:].values))
        coefs  = dict(zip(FEATURES, R['ols'].params[1:].values))
        hyps = [
            ("H1", "Branch density increases bank usage",    "BranchDensity"),
            ("H2", "Mobile ownership drives inclusion",       "MobileOwn"),
            ("H3", "Child marriage reduces inclusion",        "ChildMarriage"),
            ("H4", "Paid employment improves inclusion",      "PaidCash"),
        ]
        for code, text, feat in hyps:
            p = pvals[feat]; c = coefs[feat]
            sup = p < 0.05
            color_border = '#166534' if sup else '#7f1d1d'
            color_bg     = '#052e16' if sup else '#1a0000'
            color_text   = '#4ade80' if sup else '#f87171'
            verdict      = 'Supported (p<0.05)' if sup else f'Not sig. (p={p:.2f})'
            st.markdown(f"""
            <div style="display:flex;gap:10px;align-items:flex-start;
                        background:{color_bg};border:1px solid {color_border};
                        border-radius:8px;padding:0.7rem 0.9rem;margin-bottom:5px;">
              <span style="color:#38bdf8;font-weight:600;font-size:.78rem;
                           font-family:'DM Mono',monospace;min-width:24px">{code}</span>
              <span style="color:#94a3b8;font-size:.78rem;flex:1;line-height:1.4">{text}</span>
              <span style="color:{color_text};font-size:.7rem;font-weight:500;
                           white-space:nowrap">{verdict}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head" style="margin-top:1.5rem">Model Performance Summary</div>',
                    unsafe_allow_html=True)
        for label, r2, rmse, best in [
            ("OLS Regression",       R['r1'], R['e1'], False),
            ("Random Forest",        R['r2'], R['e2'], False),
            ("Ridge Regression",     R['r3'], R['e3'], False),
            ("Gradient Boosting",    R['r4'], R['e4'], False),
            ("Ensemble (Best)",      R['re'], R['ee'], True),
        ]:
            st.markdown(f"""
            <div class="model-row {'model-best' if best else ''}">
              <span class="model-name">{'[Best] ' if best else ''}{label}</span>
              <span class="model-val">R² {r2:.3f}</span>
              <span style="color:#818cf8;font-family:'DM Mono',monospace;font-size:.78rem">
                RMSE {rmse:.2f}</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD DATA & PREDICT
# ════════════════════════════════════════════════════════════════════════════
elif page == "Upload Data & Predict":
    st.markdown('<div class="sec-head">Upload Dataset for Batch Prediction</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        <strong>How this works:</strong> Upload your dataset as an Excel or CSV file.
        The model will automatically predict bank account usage (%) for every row
        in your file based on the four key variables. Results are shown in a table
        that you can download. This allows future researchers to apply this model
        to any new dataset.
    </div>""", unsafe_allow_html=True)

    # Template / column guide
    with st.expander("Required column names — expand to see format"):
        st.markdown("""<table class="req-table">
          <tr><th>Column Name</th><th>Description</th><th>Example Value</th></tr>
          <tr><td>BankAccount</td><td>Women with bank account they actively use (%) — optional, used for comparison if available</td><td>76.21</td></tr>
          <tr><td>BranchDensity</td><td>Bank offices per 1,000 people in that region</td><td>0.0598</td></tr>
          <tr><td>ChildMarriage</td><td>Women married before age 18 (%)</td><td>43.36</td></tr>
          <tr><td>MobileOwn</td><td>Women with a mobile phone they use (%)</td><td>49.26</td></tr>
          <tr><td>PaidCash</td><td>Women who worked and were paid in cash (%)</td><td>12.80</td></tr>
        </table>""", unsafe_allow_html=True)

        # Download template
        template_df = pd.DataFrame({
            'State/District': ['Example Region 1', 'Example Region 2'],
            'BranchDensity':  [0.12, 0.08],
            'ChildMarriage':  [25.0, 38.0],
            'MobileOwn':      [52.0, 40.0],
            'PaidCash':       [22.0, 15.0],
        })
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=csv_template,
            file_name="financial_inclusion_template.csv",
            mime="text/csv"
        )

    # Upload widget
    uploaded = st.file_uploader(
        "Upload your file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="The file must have the required column names listed above"
    )

    if uploaded is not None:
        # Read file
        try:
            if uploaded.name.endswith('.csv'):
                user_df = pd.read_csv(uploaded)
            else:
                user_df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.markdown(f"""<div class="info-box">
            File loaded: <strong>{uploaded.name}</strong> |
            Rows: <strong>{len(user_df)}</strong> |
            Columns: <strong>{len(user_df.columns)}</strong>
        </div>""", unsafe_allow_html=True)

        # Check required columns
        missing_cols = [c for c in FEATURES if c not in user_df.columns]
        if missing_cols:
            st.markdown(f"""<div class="warn-box">
                Missing required columns: <strong>{', '.join(missing_cols)}</strong><br>
                Please rename your columns to match the required names above and re-upload.
            </div>""", unsafe_allow_html=True)
        else:
            # Drop rows with missing values in required columns
            clean_df = user_df.dropna(subset=FEATURES).copy()
            dropped  = len(user_df) - len(clean_df)
            if dropped > 0:
                st.markdown(f"""<div class="warn-box">
                    {dropped} row(s) skipped due to missing values in required columns.
                </div>""", unsafe_allow_html=True)

            if len(clean_df) == 0:
                st.error("No valid rows found after removing missing values.")
                st.stop()

            # Run predictions
            results = []
            for _, row in clean_df.iterrows():
                inp = [row['BranchDensity'], row['ChildMarriage'],
                       row['MobileOwn'],     row['PaidCash']]
                p1, p2, p3, p4, ens = predict_row(inp)
                results.append({'OLS': p1, 'RF': p2, 'Ridge': p3, 'GBM': p4, 'Ensemble': ens})

            pred_df = pd.DataFrame(results)

            # Add predictions to df
            out_df = clean_df.copy().reset_index(drop=True)
            out_df['Pred_OLS']      = pred_df['OLS'].round(2)
            out_df['Pred_RF']       = pred_df['RF'].round(2)
            out_df['Pred_Ridge']    = pred_df['Ridge'].round(2)
            out_df['Pred_GBM']      = pred_df['GBM'].round(2)
            out_df['Pred_Ensemble'] = pred_df['Ensemble'].round(2)

            if TARGET in out_df.columns:
                out_df['Error'] = (out_df['Pred_Ensemble'] - out_df[TARGET]).round(2)

            st.markdown('<div class="sec-head" style="margin-top:1.5rem">Prediction Results</div>',
                        unsafe_allow_html=True)

            # Summary metrics
            avg_pred = pred_df['Ensemble'].mean()
            max_pred = pred_df['Ensemble'].max()
            min_pred = pred_df['Ensemble'].min()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows Predicted", len(clean_df))
            c2.metric("Average Predicted", f"{avg_pred:.1f}%")
            c3.metric("Highest Predicted", f"{max_pred:.1f}%")
            c4.metric("Lowest Predicted",  f"{min_pred:.1f}%")

            # Show results
            state_col = None
            for possible in ['State', 'District', 'Region', 'State/District',
                             'State/UT', 'Name', 'state', 'district']:
                if possible in out_df.columns:
                    state_col = possible
                    break

            st.markdown("**Row-by-row predictions (Ensemble model):**")
            for i, row in out_df.iterrows():
                name  = str(row[state_col]) if state_col else f"Row {i+1}"
                ens   = row['Pred_Ensemble']
                gap   = ens - nat_avg
                arrow = "+" if gap >= 0 else ""
                gap_color = 'result-delta-pos' if gap >= 0 else 'result-delta-neg'
                actual_str = ""
                if TARGET in row.index and pd.notna(row.get(TARGET)):
                    actual_str = f"&nbsp;&nbsp;|&nbsp;&nbsp;Actual: {row[TARGET]:.1f}%"
                st.markdown(f"""
                <div class="result-row">
                  <span class="result-state">{name}</span>
                  <span>
                    <span class="result-val">{ens:.1f}%</span>
                    &nbsp;&nbsp;
                    <span class="{gap_color}">{arrow}{gap:.1f}% vs national avg</span>
                    {actual_str}
                  </span>
                </div>""", unsafe_allow_html=True)

            # Visualise predictions
            st.markdown('<div class="sec-head" style="margin-top:1.5rem">Prediction Chart</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, max(4, len(out_df) * 0.4)))
            names = [str(out_df[state_col].iloc[i]) if state_col else f"Row {i+1}"
                     for i in range(len(out_df))]
            vals  = out_df['Pred_Ensemble'].values
            bar_c = [ACCENT3 if v >= 85 else ACCENT if v >= 75 else DANGER for v in vals]
            ax.barh(names, vals, color=bar_c, edgecolor='none', height=0.6)
            ax.axvline(nat_avg, color=AMBER, linestyle='--', linewidth=1.2,
                       label=f'National avg: {nat_avg:.1f}%')
            if TARGET in out_df.columns:
                ax.scatter(out_df[TARGET].values,
                           range(len(out_df)),
                           color='white', s=40, zorder=5, label='Actual value')
            for i, v in enumerate(vals):
                ax.text(v + 0.3, i, f'{v:.1f}', va='center', fontsize=7.5, color='#94a3b8')
            ax.set_xlabel('Bank Account Usage Prediction (%)')
            ax.set_title('Ensemble Model — Predicted Bank Account Usage')
            ax.legend(facecolor=DARK, edgecolor='#1e3a52', labelcolor='#94a3b8', fontsize=8)
            ax.grid(axis='x', alpha=0.3)
            fig.tight_layout(); st.pyplot(fig); plt.close()

            # Download results
            st.markdown("**Download results:**")
            csv_out = out_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_out,
                file_name="financial_inclusion_predictions.csv",
                mime="text/csv"
            )

    else:
        st.markdown("""
        <div class="upload-zone">
          <div class="upload-title">Upload your CSV or Excel file above</div>
          <div class="upload-sub">The file must contain the required columns listed in the guide above.<br>
          Download the template to get started with the correct format.</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MANUAL PREDICT
# ════════════════════════════════════════════════════════════════════════════
elif page == "Manual Predict":
    st.markdown('<div class="sec-head">Manual Prediction — Single Region Scenario</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        Enter values manually to predict bank account usage for any region.
        Use the scenario buttons to instantly see the impact of policy changes.
    </div>""", unsafe_allow_html=True)

    col_s, col_r = st.columns([1, 1])

    with col_s:
        st.markdown("**Set Region Values**")
        branch = st.slider("Branch Density (bank offices per 1,000 people)",
                           0.05, 0.50, 0.13, 0.01)
        child  = st.slider("Child Marriage Rate (%)", 0.0, 55.0, 20.0, 0.5)
        mobile = st.slider("Mobile Phone Ownership (%)", 20.0, 100.0, 55.0, 0.5)
        paid   = st.slider("Women Paid in Cash (%)", 10.0, 60.0, 25.0, 0.5)

    with col_r:
        st.markdown("**Prediction**")
        p1, p2, p3, p4, ens = predict_row([branch, child, mobile, paid])
        gap = ens - nat_avg
        arrow = "+" if gap >= 0 else ""
        color = "#4ade80" if gap >= 0 else "#f87171"

        st.markdown(f"""
        <div class="pred-card">
          <div class="pred-label">Predicted Bank Account Usage</div>
          <div class="pred-main">{ens:.1f}%</div>
          <div class="pred-delta" style="color:{color}">
            {arrow}{gap:.1f}% vs national average ({nat_avg:.1f}%)
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Model Breakdown**")
        for label, val, best in [
            ("OLS Regression",    p1,  False),
            ("Random Forest",     p2,  False),
            ("Ridge Regression",  p3,  False),
            ("Gradient Boosting", p4,  False),
            ("Ensemble",          ens, True),
        ]:
            st.markdown(f"""
            <div class="model-row {'model-best' if best else ''}">
              <span class="model-name">{'[Best] ' if best else ''}{label}</span>
              <span class="model-val">{val:.2f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("**Policy Scenario — What If?**")
        w1, w2, w3 = st.columns(3)

        def pred_new(b, c, m, p):
            _, _, _, _, e = predict_row([b, c, m, p])
            return e

        with w1:
            if st.button("Branch Density +0.1", use_container_width=True):
                new = pred_new(min(0.5, branch+0.1), child, mobile, paid)
                st.metric("New Prediction", f"{new:.1f}%", f"{new-ens:+.2f}%")
        with w2:
            if st.button("Child Marriage -10%", use_container_width=True):
                new = pred_new(branch, max(0, child-10), mobile, paid)
                st.metric("New Prediction", f"{new:.1f}%", f"{new-ens:+.2f}%")
        with w3:
            if st.button("Mobile +15%", use_container_width=True):
                new = pred_new(branch, child, min(100, mobile+15), paid)
                st.metric("New Prediction", f"{new:.1f}%", f"{new-ens:+.2f}%")

        st.markdown("**Recommended Interventions**")
        recs = []
        if branch < 0.10: recs.append("Branch density is critically low — expand Business Correspondent agent network")
        if child   > 30:  recs.append("Child marriage rate is high — enforce legal protections and education programs")
        if mobile  < 45:  recs.append("Mobile ownership is low — targeted smartphone access schemes recommended")
        if paid    < 18:  recs.append("Low paid employment — expand MGNREGA and rural livelihood programs")
        if not recs:
            st.markdown('<div class="ok-box">All indicators are at acceptable levels. Maintain current programs.</div>',
                        unsafe_allow_html=True)
        else:
            for rec in recs:
                st.markdown(f'<div class="warn-box">{rec}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Analysis":
    st.markdown('<div class="sec-head">Variable Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Scatter Analysis", "OLS Summary"])

    with tab1:
        cols_p  = ['BankAccount'] + FEATURES
        labels_p = ['Bank Account %', 'Branch Density', 'Child Marriage %',
                    'Mobile Ownership %', 'Women Paid in Cash %']
        cdf = ref_df[cols_p].copy(); cdf.columns = labels_p
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cdf.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, linewidths=0.5,
                    annot_kws={'size': 9, 'color': '#e2e8f0'},
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Pearson Correlation Matrix', pad=12)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Correlation with Bank Account Usage:**")
        cv_ = cdf.corr()['Bank Account %'].drop('Bank Account %').sort_values(ascending=False)
        for var, val in cv_.items():
            color = ACCENT if val > 0 else DANGER
            bw    = int(abs(val) * 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:5px 0">
              <span style="color:#94a3b8;font-size:.78rem;min-width:175px">{var}</span>
              <div style="flex:1;background:#1e3a52;border-radius:4px;height:7px;overflow:hidden">
                <div style="width:{bw}%;background:{color};height:100%;border-radius:4px"></div>
              </div>
              <span style="font-family:'DM Mono',monospace;font-size:.78rem;color:{color};
                           min-width:50px;text-align:right">{val:+.3f}</span>
            </div>""", unsafe_allow_html=True)

    with tab2:
        fs = st.selectbox("Select variable", FEATURES, format_func=lambda x: LABELS[x])
        fig, ax = plt.subplots(figsize=(9, 5))
        sc_ = ax.scatter(ref_df[fs], ref_df[TARGET],
                         c=ref_df[TARGET], cmap='cool',
                         s=90, alpha=0.9, edgecolors='#1e3a52', linewidth=0.8, zorder=3)
        z  = np.polyfit(ref_df[fs], ref_df[TARGET], 1)
        xl = np.linspace(ref_df[fs].min(), ref_df[fs].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color=AMBER, linewidth=1.5,
                linestyle='--', label='Trend line', zorder=2)
        for _, row in ref_df.iterrows():
            ax.annotate(row['State'][:7], (row[fs], row[TARGET]),
                        fontsize=5.5, color='#64748b', ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points')
        plt.colorbar(sc_, ax=ax, label='Bank Account %')
        ax.set_xlabel(LABELS[fs]); ax.set_ylabel('Bank Account Usage (%)')
        ax.set_title(f'{LABELS[fs]} vs Bank Account Usage')
        ax.legend(facecolor=DARK, edgecolor='#1e3a52', labelcolor='#94a3b8', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        r = ref_df[[fs, TARGET]].corr().iloc[0, 1]
        direction = "positive" if r > 0 else "negative"
        strength  = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
        st.markdown(f"""<div class="info-box">
            Pearson r = <strong>{r:.3f}</strong> —
            <strong>{strength} {direction}</strong> relationship.
            {'Higher ' + LABELS[fs] + ' is associated with higher bank account usage.'
             if r > 0 else 'Higher ' + LABELS[fs] + ' is associated with lower bank account usage.'}
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("**Full OLS Regression Summary**")
        st.text(str(R['ols'].summary()))
        st.markdown(f"""<div class="info-box">
            <strong>Key Finding:</strong> BranchDensity is the only statistically significant
            predictor (p = {R['ols'].pvalues[1]:.3f} &lt; 0.05). Coefficient = +{R['ols'].params[1]:.2f},
            meaning each additional bank branch per 1,000 people is associated with approximately
            {R['ols'].params[1]:.1f}% increase in rural women's bank account usage.
            R² = {R['ols'].rsquared:.4f} | Adjusted R² = {R['ols'].rsquared_adj:.4f}.
            The limited sample size (n=33 states) reduces statistical power for other variables.
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="sec-head">Model Performance — Leave-One-Out Cross Validation</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
        <strong>Why Leave-One-Out (LOO)?</strong> With only 33 state-level observations,
        standard train-test split is unreliable. LOO uses each state as a test point once
        while training on all others — giving the most honest performance estimate for small datasets.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.1])

    with col1:
        st.markdown("**Performance Metrics (LOO CV)**")
        for label, r2, rmse, best in [
            ("OLS Regression",    R['r1'], R['e1'], False),
            ("Random Forest",     R['r2'], R['e2'], False),
            ("Ridge Regression",  R['r3'], R['e3'], False),
            ("Gradient Boosting", R['r4'], R['e4'], False),
            ("Ensemble",          R['re'], R['ee'], True),
        ]:
            st.markdown(f"""
            <div class="model-row {'model-best' if best else ''}">
              <span class="model-name">{'[Best] ' if best else ''}{label}</span>
              <span class="model-val">R² {r2:.4f}</span>
              <span style="color:#818cf8;font-family:'DM Mono',monospace;font-size:.78rem">
                RMSE {rmse:.3f}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="info-box" style="margin-top:1rem">
            <strong>RMSE interpretation:</strong> RMSE is in percentage points.
            An RMSE of 5 means predictions are off by approximately 5 percentage
            points on average — acceptable for a social policy model with n=33.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**Actual vs Ensemble Predicted (LOO)**")
        fig, ax = plt.subplots(figsize=(6, 5))
        y_arr = R['y']
        p_arr = R['p_ens']
        ax.scatter(y_arr, p_arr, color=ACCENT, s=80,
                   edgecolors='#1e3a52', linewidth=0.8, alpha=0.9, zorder=3)
        mn = min(float(min(y_arr)), float(min(p_arr))) - 2
        mx = max(float(max(y_arr)), float(max(p_arr))) + 2
        ax.plot([mn, mx], [mn, mx], color=AMBER, linestyle='--', linewidth=1.5,
                label='Perfect prediction', zorder=2)
        ax.set_xlabel('Actual Bank Account Usage (%)')
        ax.set_ylabel('Predicted Bank Account Usage (%)')
        ax.set_title('Actual vs Predicted — Ensemble LOO')
        ax.legend(facecolor=DARK, edgecolor='#1e3a52', labelcolor='#94a3b8', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Residuals — Prediction Error by State**")
        res_df = ref_df[['State']].copy()
        res_df['Actual']    = R['y']
        res_df['Predicted'] = np.round(R['p_ens'], 2)
        res_df['Error']     = (res_df['Predicted'] - res_df['Actual']).round(2)
        res_df = res_df.sort_values('Error')
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        colors_ = [DANGER if v < 0 else ACCENT3 for v in res_df['Error']]
        ax2.barh(res_df['State'], res_df['Error'], color=colors_, edgecolor='none', height=0.65)
        ax2.axvline(0, color='white', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('Prediction Error (Predicted - Actual %)')
        ax2.set_title('Residuals — Ensemble Model')
        ax2.tick_params(axis='y', labelsize=7)
        ax2.grid(axis='x', alpha=0.3)
        fig2.tight_layout(); st.pyplot(fig2); plt.close()
