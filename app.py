# ============================================================
# Financial Inclusion of Rural Women in India — ABA Project
# Pages: Dashboard | Predict | Analysis
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="FinInclude India", page_icon="\U0001f3e6",
                   layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2rem 2rem 2rem;max-width:1400px;}
.topbar{background:#0f1923;padding:1rem 2rem;margin:-1rem -2rem 2rem -2rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1e3a52;}
.topbar-logo{color:#38bdf8;font-weight:600;font-size:1.15rem;}
.topbar-sub{color:#64748b;font-size:0.75rem;margin-top:2px;}
.topbar-badge{background:#1e3a52;color:#38bdf8;font-size:0.7rem;padding:3px 12px;border-radius:99px;font-weight:500;border:1px solid #38bdf833;}
.kpi{background:#0f1923;border:1px solid #1e3a52;border-radius:12px;padding:1.2rem 1.4rem;position:relative;overflow:hidden;}
.kpi::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#38bdf8,#818cf8);}
.kpi-lbl{color:#64748b;font-size:.7rem;font-weight:500;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem;}
.kpi-val{color:#f1f5f9;font-size:1.9rem;font-weight:600;letter-spacing:-0.03em;line-height:1.1;}
.kpi-sub{color:#38bdf8;font-size:.72rem;margin-top:.3rem;}
.sec{color:#f1f5f9;font-size:.95rem;font-weight:600;margin-bottom:.8rem;padding-bottom:.4rem;border-bottom:1px solid #1e293b;}
.card{background:#0d2137;border:1px solid #1e3a52;border-left:3px solid #38bdf8;border-radius:0 8px 8px 0;padding:.7rem 1rem;color:#94a3b8;font-size:.83rem;line-height:1.6;margin:.4rem 0;}
.card strong{color:#e2e8f0;}
.pred-box{background:#0d2137;border:1px solid #38bdf844;border-radius:16px;padding:2rem;text-align:center;margin:1rem 0;}
.pred-num{color:#38bdf8;font-size:3.5rem;font-weight:700;}
.pred-lbl{color:#64748b;font-size:.82rem;margin-top:.25rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

BG="#0f1923"; C1="#38bdf8"; C2="#818cf8"; C3="#4ade80"; C4="#f87171"; AM="#f59e0b"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"axes.edgecolor":"#1e3a52",
    "axes.labelcolor":"#94a3b8","xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#e2e8f0","grid.color":"#1e3a52","grid.linestyle":"--","grid.alpha":.5,
    "font.family":"sans-serif","axes.titlecolor":"#f1f5f9","axes.titlesize":10,
    "axes.titleweight":"bold","axes.labelsize":8.5,"figure.dpi":130})

@st.cache_data
def load():
    return pd.DataFrame({
        "State":["Andaman & Nicobar","Andhra Pradesh","Arunachal Pradesh","Assam","Bihar",
                 "Chhattisgarh","Dadra & NH + DD","Goa","Gujarat","Haryana","Himachal Pradesh",
                 "Jammu & Kashmir","Jharkhand","Karnataka","Kerala","Ladakh","Madhya Pradesh",
                 "Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Puducherry",
                 "Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
                 "Uttar Pradesh","Uttarakhand","West Bengal"],
        "BankAccount":[89.75,79.63,76.74,77.86,76.21,81.14,89.28,92.38,67.48,72.39,82.23,
                       83.52,79.75,87.68,78.22,88.65,73.34,70.91,70.71,68.22,74.97,55.41,
                       87.38,96.72,82.10,79.00,76.70,91.72,85.21,77.67,74.06,79.75,73.24],
        "BranchDensity":[0.1703,0.1345,0.1045,0.0830,0.0598,0.0959,0.1230,0.4529,0.1225,0.1770,
                         0.2238,0.1296,0.0821,0.1595,0.1936,0.2345,0.0847,0.1112,0.0591,0.1057,
                         0.1638,0.0819,0.1176,0.1576,0.2234,0.0978,0.2414,0.1548,0.1448,0.1355,
                         0.0802,0.1918,0.0955],
        "HHDecision":[95.52,84.30,86.64,91.77,87.02,91.54,87.43,98.60,90.67,86.15,93.90,81.66,
                      89.84,80.52,94.61,80.27,84.13,89.15,95.00,91.98,98.02,99.82,90.32,96.55,
                      90.25,86.79,93.94,93.65,86.16,89.47,86.78,90.64,85.81],
        "ChildMarriage":[15.27,32.90,19.33,33.35,43.36,13.20,26.16,3.18,26.91,13.65,5.14,5.26,
                         36.09,24.66,8.22,3.13,26.55,27.57,17.61,19.12,14.00,7.25,21.71,0.97,
                         8.68,28.28,12.54,15.18,27.38,42.40,17.85,9.76,48.13],
        "Literacy":[85.58,63.84,71.63,75.39,54.54,71.30,67.90,93.43,69.03,78.75,91.23,74.71,
                    59.34,70.95,97.47,76.60,63.30,79.51,84.84,85.53,87.67,82.74,69.22,89.07,
                    80.04,61.63,86.24,81.50,58.10,76.88,65.55,79.78,72.54],
        "MobileOwn":[80.86,40.93,75.28,53.87,49.26,33.94,45.53,87.06,36.18,43.41,77.77,73.25,
                     43.67,53.42,86.90,81.22,31.44,43.07,68.19,64.29,70.64,76.33,47.95,76.56,
                     54.90,45.30,83.25,68.93,50.60,48.03,42.37,55.65,39.07],
        "InternetUse":[27.87,15.35,49.61,24.36,17.00,20.78,23.84,68.28,17.53,42.80,45.23,38.90,
                       22.69,24.76,57.54,53.98,20.08,23.71,40.39,27.95,47.98,40.26,21.33,50.37,
                       48.75,30.76,68.08,39.20,15.78,17.66,24.48,39.35,13.98],
        "PaidCash":[17.12,44.50,23.55,19.27,12.80,42.62,39.47,27.86,34.05,17.03,17.71,18.50,
                    17.74,41.36,25.79,28.25,28.00,39.60,44.03,39.11,29.00,20.67,25.58,44.84,
                    20.21,17.52,29.25,45.51,55.54,25.93,14.83,22.22,20.17],
        "AssetOwn":[19.36,50.63,70.25,43.85,55.67,45.45,59.74,24.07,43.27,41.03,23.38,60.77,
                    66.47,69.65,29.18,72.95,41.27,24.50,58.93,70.11,28.36,28.86,45.47,38.30,
                    67.07,26.61,50.57,52.04,74.46,17.28,53.53,25.13,22.46],
    })

df = load()
FEATS3 = ["BranchDensity","HHDecision","ChildMarriage"]
VLABELS = {"BankAccount":"Bank Account Usage %","BranchDensity":"Branch Density (per 1K)",
           "HHDecision":"HH Decision Making %","ChildMarriage":"Child Marriage %",
           "Literacy":"Female Literacy %","MobileOwn":"Mobile Ownership %",
           "InternetUse":"Internet Usage %","PaidCash":"Women Paid in Cash %","AssetOwn":"Asset Ownership %"}
OLS = {"intercept":110.411,"BranchDensity":56.3041,"HHDecision":-0.4251,"ChildMarriage":-0.0479}

@st.cache_resource
def train():
    X=df[FEATS3].values; y=df["BankAccount"].values
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    rf=RandomForestRegressor(n_estimators=300,max_depth=4,random_state=42).fit(Xs,y)
    gb=GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,random_state=42).fit(Xs,y)
    op=(OLS["intercept"]+OLS["BranchDensity"]*df["BranchDensity"]+
        OLS["HHDecision"]*df["HHDecision"]+OLS["ChildMarriage"]*df["ChildMarriage"]).values
    ens=(op+rf.predict(Xs)+gb.predict(Xs))/3
    return sc,rf,gb,op,ens,y

sc,rf,gb,ols_pred,ens_pred,y_arr = train()

def predict_fn(bd,hh,cm):
    Xs=sc.transform([[bd,hh,cm]])
    ols_val=OLS["intercept"]+OLS["BranchDensity"]*bd+OLS["HHDecision"]*hh+OLS["ChildMarriage"]*cm
    p=(ols_val+rf.predict(Xs)[0]+gb.predict(Xs)[0])/3
    return round(float(np.clip(p,40,100)),1), round(ols_val,1)

def bar_row(label, val, navg, cc, maxv, is_density=False):
    diff=val-navg; bw=int((val/maxv)*100) if maxv>0 else 0
    fmt=f"{val:.4f}" if is_density else f"{val:.1f}"
    unit="/1K" if is_density else "%"
    return f"""<div style="margin:9px 0">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="color:#94a3b8;font-size:.79rem">{label}</span>
        <span style="font-family:'DM Mono',monospace;font-size:.79rem;color:{cc}">{fmt}{unit}&nbsp;({diff:+.2f})</span>
      </div>
      <div style="background:#1e3a52;border-radius:4px;height:7px;overflow:hidden">
        <div style="width:{bw}%;background:{cc};height:100%;border-radius:4px"></div>
      </div></div>"""

st.markdown("""<div class="topbar">
  <div><div class="topbar-logo">\U0001f3e6 FinInclude India</div>
  <div class="topbar-sub">Financial Inclusion of Rural Women &middot; NFHS-5 (2019&ndash;21) + RBI (2021) &middot; ABA Final Project</div></div>
  <span class="topbar-badge">33 States / UTs</span>
</div>""", unsafe_allow_html=True)

page=st.radio("",["\U0001f4ca Dashboard","\U0001f3af Predict","\U0001f4c8 Analysis"],
              horizontal=True,label_visibility="collapsed")
st.markdown("<hr style='border-color:#1e3a52;margin:.2rem 0 1.5rem'>",unsafe_allow_html=True)

# ═══ DASHBOARD ════════════════════════════════════════════
if "Dashboard" in page:
    states=sorted(df["State"].tolist())
    sel=st.selectbox("\U0001f5fa Select a State / UT",states,index=states.index("Tamil Nadu"))
    row=df[df["State"]==sel].iloc[0]
    avg=df.mean(numeric_only=True)
    rank=int(df["BankAccount"].rank(ascending=False)[df["State"]==sel].values[0])
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    kpi_data=[
        (c1,"Bank Account Usage",f"{row['BankAccount']:.1f}%",f"Rank #{rank} of 33"),
        (c2,"Branch Density",f"{row['BranchDensity']:.4f}/1K",f"{row['BranchDensity']-avg['BranchDensity']:+.4f} vs avg"),
        (c3,"HH Decision Making",f"{row['HHDecision']:.1f}%",f"{row['HHDecision']-avg['HHDecision']:+.1f}% vs avg"),
        (c4,"Child Marriage",f"{row['ChildMarriage']:.1f}%",f"{row['ChildMarriage']-avg['ChildMarriage']:+.1f}% vs avg"),
    ]
    for col,lbl,val,sub in kpi_data:
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                        f'<div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>',
                        unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    left,right=st.columns([1.2,1])
    with left:
        st.markdown(f'<div class="sec">{sel} — All Indicators vs National Average</div>',unsafe_allow_html=True)
        for var in ["BankAccount","BranchDensity","HHDecision","ChildMarriage",
                    "Literacy","MobileOwn","InternetUse","PaidCash","AssetOwn"]:
            v=row[var]; na=avg[var]; diff=v-na
            better=(diff>=0) if var!="ChildMarriage" else (diff<=0)
            cc=C3 if better else C4
            maxv=df[var].max()
            st.markdown(bar_row(VLABELS[var],v,na,cc,maxv,var=="BranchDensity"),unsafe_allow_html=True)
    with right:
        st.markdown('<div class="sec">All States — Bank Account Usage %</div>',unsafe_allow_html=True)
        ranked=df.sort_values("BankAccount",ascending=True)
        fig,ax=plt.subplots(figsize=(5.5,9))
        colors=[AM if s==sel else (C3 if v>=avg["BankAccount"] else C1)
                for s,v in zip(ranked["State"],ranked["BankAccount"])]
        bars=ax.barh(ranked["State"],ranked["BankAccount"],color=colors,edgecolor="none",height=0.7)
        ax.axvline(avg["BankAccount"],color=C2,linestyle="--",lw=1.5,label=f"Avg {avg['BankAccount']:.1f}%")
        for bar,v in zip(bars,ranked["BankAccount"]):
            ax.text(v+0.3,bar.get_y()+bar.get_height()/2,f"{v:.0f}",va="center",fontsize=7,color="#94a3b8")
        ax.set_xlabel("Bank Account Usage %"); ax.set_xlim(45,108); ax.set_title(f"Highlighted: {sel}",fontsize=9)
        ax.legend(facecolor=BG,edgecolor="#1e3a52",labelcolor="#94a3b8",fontsize=8)
        ax.grid(axis="x",alpha=0.3); fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="sec" style="margin-top:1.5rem">National Overview</div>',unsafe_allow_html=True)
    n1,n2,n3,n4=st.columns(4)
    for col,lbl,val in [
        (n1,"National Avg",f"{avg['BankAccount']:.1f}%"),
        (n2,"Highest",f"{df.loc[df['BankAccount'].idxmax(),'State']} ({df['BankAccount'].max():.1f}%)"),
        (n3,"Lowest",f"{df.loc[df['BankAccount'].idxmin(),'State']} ({df['BankAccount'].min():.1f}%)"),
        (n4,"Above Average",f"{(df['BankAccount']>avg['BankAccount']).sum()} / 33 states"),
    ]:
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-lbl">{lbl}</div>'
                        f'<div class="kpi-val" style="font-size:1.2rem">{val}</div></div>',
                        unsafe_allow_html=True)

# ═══ PREDICT ═════════════════════════════════════════════
elif "Predict" in page:
    st.markdown('<div class="sec">What-If Scenario Predictor</div>',unsafe_allow_html=True)
    st.markdown('<div class="card">Select a state to auto-fill or adjust sliders. '
                'The ensemble predicts <strong>bank account usage %</strong> among rural women.</div>',
                unsafe_allow_html=True)
    col_in,col_out=st.columns([1.1,1])
    with col_in:
        st.markdown("**Select state to auto-fill:**")
        chosen=st.selectbox("",["— Manual —"]+sorted(df["State"].tolist()),label_visibility="collapsed")
        if chosen!="— Manual —":
            r=df[df["State"]==chosen].iloc[0]
            dbd,dhh,dcm=float(r["BranchDensity"]),float(r["HHDecision"]),float(r["ChildMarriage"])
        else:
            dbd=float(df["BranchDensity"].mean())
            dhh=float(df["HHDecision"].mean())
            dcm=float(df["ChildMarriage"].mean())
        st.markdown("<br>**Adjust the indicators:**",unsafe_allow_html=True)
        bd=st.slider("\U0001f3e6 Branch Density (per 1,000 people)",0.04,0.50,dbd,0.005,format="%.3f")
        hh=st.slider("\U0001f3e0 Household Decision Making (%)",75.0,100.0,dhh,0.5)
        cm=st.slider("\U0001f467 Child Marriage Rate (%)",0.0,55.0,dcm,0.5)
        st.markdown("<br>**Quick presets:**",unsafe_allow_html=True)
        b1,b2,b3=st.columns(3)
        if b1.button("\U0001f4cd National Avg"):
            bd=float(df["BranchDensity"].mean()); hh=float(df["HHDecision"].mean()); cm=float(df["ChildMarriage"].mean())
        if b2.button("\U0001f31f Best (Goa)"):
            bd=0.4529; hh=98.60; cm=3.18
        if b3.button("\u26a0 Low Inclusion"):
            bd=0.0598; hh=87.02; cm=43.36
    with col_out:
        pred,ols_p=predict_fn(bd,hh,cm)
        nat=float(df["BankAccount"].mean()); diff=pred-nat
        dc=C3 if diff>=0 else C4; arrow="\u25b2" if diff>=0 else "\u25bc"
        st.markdown(f'<div class="pred-box">'
                    f'<div class="pred-lbl">Predicted Bank Account Usage</div>'
                    f'<div class="pred-num">{pred}%</div>'
                    f'<div style="color:{dc};font-size:1rem;margin-top:.5rem;font-weight:600">'
                    f'{arrow} {abs(diff):.1f}% vs national average ({nat:.1f}%)</div>'
                    f'<div class="pred-lbl" style="margin-top:.8rem">Ensemble: OLS + Random Forest + Gradient Boosting</div>'
                    f'</div>',unsafe_allow_html=True)
        st.markdown("**Variable contributions (OLS):**")
        contribs={"\U0001f3e6 Branch Density":OLS["BranchDensity"]*bd,
                  "\U0001f3e0 HH Decision Making":OLS["HHDecision"]*hh,
                  "\U0001f467 Child Marriage":OLS["ChildMarriage"]*cm}
        for lbl,contrib in sorted(contribs.items(),key=lambda x:abs(x[1]),reverse=True):
            cc=C3 if contrib>0 else C4
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                        f'border-bottom:1px solid #1e293b">'
                        f'<span style="color:#94a3b8;font-size:.8rem">{lbl}</span>'
                        f'<span style="color:{cc};font-family:DM Mono,monospace;font-size:.82rem">{contrib:+.2f}</span>'
                        f'</div>',unsafe_allow_html=True)
        st.markdown("<br>**Inclusion zone:**",unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(5,1.5))
        zones=[(55,"Low\n<65%",C4),(65,"Moderate\n65-75%",AM),(75,"Good\n75-85%",C1),(85,"High\n>85%",C3)]
        ends=[65,75,85,100]
        for i,(start,lbl,color) in enumerate(zones):
            ax.barh(0,ends[i]-start,left=start,color=color,height=0.5,edgecolor="none",alpha=0.75)
            ax.text((start+ends[i])/2,0,lbl,ha="center",va="center",fontsize=6.5,color="white",fontweight="bold")
        ax.axvline(pred,color="white",lw=2.5,zorder=5)
        ax.text(pred,0.32,f"{pred}%",ha="center",fontsize=9,color="white",fontweight="bold")
        ax.set_xlim(50,102);ax.set_ylim(-0.4,0.6);ax.axis("off");fig.tight_layout()
        st.pyplot(fig);plt.close()

# ═══ ANALYSIS ════════════════════════════════════════════
elif "Analysis" in page:
    st.markdown('<div class="sec">Data Insights & Model Performance</div>',unsafe_allow_html=True)
    col1,col2=st.columns([1,1.05])
    with col1:
        st.markdown("**Key Findings**")
        for icon,title,text in [
            ("\U0001f3e6","Branch Density","The strongest driver of rural women's financial inclusion. States with more bank offices per 1,000 people show consistently higher account usage."),
            ("\U0001f3e0","Household Decision Making","Women who participate in financial decisions are more likely to hold and actively use bank accounts. Autonomy enables inclusion."),
            ("\U0001f467","Child Marriage","Higher child marriage rates correlate with lower financial inclusion. Early marriage interrupts education and economic participation."),
        ]:
            st.markdown(f'<div style="background:#0f1923;border:1px solid #1e3a52;border-radius:10px;'
                        f'padding:.85rem 1rem;margin-bottom:.5rem">'
                        f'<div style="color:#38bdf8;font-weight:600;font-size:.85rem;margin-bottom:4px">{icon} {title}</div>'
                        f'<div style="color:#94a3b8;font-size:.8rem;line-height:1.55">{text}</div>'
                        f'</div>',unsafe_allow_html=True)
        st.markdown("<br>**Variable Importance**",unsafe_allow_html=True)
        for feat,score in sorted(zip(FEATS3,rf.feature_importances_),key=lambda x:x[1],reverse=True):
            bw=int(score*500)
            st.markdown(f'<div style="display:flex;align-items:center;gap:10px;margin:7px 0">'
                        f'<span style="color:#94a3b8;font-size:.79rem;min-width:180px">{VLABELS[feat]}</span>'
                        f'<div style="flex:1;background:#1e3a52;border-radius:4px;height:8px;overflow:hidden">'
                        f'<div style="width:{min(bw,100)}%;background:{C1};height:100%;border-radius:4px"></div></div>'
                        f'<span style="font-family:DM Mono,monospace;font-size:.78rem;color:{C1};min-width:40px;text-align:right">{score:.3f}</span>'
                        f'</div>',unsafe_allow_html=True)
        st.markdown("<br>**Model Performance**",unsafe_allow_html=True)
        ols_r2=r2_score(y_arr,ols_pred)
        rf_r2=r2_score(y_arr,rf.predict(sc.transform(df[FEATS3].values)))
        ens_r2=r2_score(y_arr,ens_pred)
        for name,r2,note in [("OLS (3 variables)",ols_r2,"Interpretable linear model"),
                              ("Random Forest",rf_r2,"Captures non-linear patterns"),
                              ("Ensemble (OLS+RF+GBM) \u2705",ens_r2,"Most reliable prediction")]:
            bg="#0d2137" if "\u2705" in name else "#0f1923"
            bc="#38bdf844" if "\u2705" in name else "#1e3a52"
            st.markdown(f'<div style="background:{bg};border:1px solid {bc};border-radius:8px;'
                        f'padding:8px 12px;margin-bottom:5px;display:flex;align-items:center;gap:10px">'
                        f'<span style="color:#e2e8f0;font-size:.82rem;flex:1">{name}</span>'
                        f'<span style="font-family:DM Mono,monospace;color:{C1};font-weight:600;font-size:.85rem">R\u00b2 {r2:.3f}</span>'
                        f'<span style="color:#64748b;font-size:.72rem;min-width:160px;text-align:right">{note}</span>'
                        f'</div>',unsafe_allow_html=True)
    with col2:
        for xcol,title in [("BranchDensity","Branch Density vs Bank Account Usage"),
                            ("ChildMarriage","Child Marriage vs Bank Account Usage")]:
            st.markdown(f"**{title}**")
            fig,ax=plt.subplots(figsize=(5.5,3.8))
            cmap="Blues" if xcol=="BranchDensity" else "Reds"
            ax.scatter(df[xcol],df["BankAccount"],c=df["BankAccount"],cmap=cmap,
                       vmin=50,vmax=100,s=75,edgecolors="#1e3a52",lw=0.8,zorder=3)
            m,b=np.polyfit(df[xcol],df["BankAccount"],1)
            xl=np.linspace(df[xcol].min(),df[xcol].max(),100)
            ax.plot(xl,m*xl+b,color=AM,lw=1.8,linestyle="--",label="Trend")
            for _,r in df.iterrows():
                cond=(r["BankAccount"]>91 or r["BankAccount"]<60 or
                      (xcol=="BranchDensity" and r[xcol]>0.35) or
                      (xcol=="ChildMarriage" and r[xcol]>40))
                if cond:
                    ax.annotate(r["State"][:8],(r[xcol],r["BankAccount"]),xytext=(4,3),
                                textcoords="offset points",fontsize=6.2,color="#64748b")
            corr=df[xcol].corr(df["BankAccount"])
            ax.set_xlabel(VLABELS[xcol]); ax.set_ylabel("Bank Account Usage %")
            ax.set_title(f"r = {corr:.2f}")
            ax.legend(facecolor=BG,edgecolor="#1e3a52",labelcolor="#94a3b8",fontsize=8)
            ax.grid(True,alpha=0.3); fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("**Actual vs Predicted (Ensemble)**")
        ens_r2=r2_score(y_arr,ens_pred)
        fig3,ax3=plt.subplots(figsize=(5.5,3.8))
        ax3.scatter(y_arr,ens_pred,color=C1,s=70,edgecolors="#1e3a52",lw=0.8,alpha=0.9,zorder=3)
        mn=min(y_arr.min(),ens_pred.min())-2; mx=max(y_arr.max(),ens_pred.max())+2
        ax3.plot([mn,mx],[mn,mx],color=AM,linestyle="--",lw=1.5,label="Perfect fit")
        ax3.set_xlabel("Actual %"); ax3.set_ylabel("Predicted %")
        ax3.set_title(f"Ensemble  R\u00b2 = {ens_r2:.3f}")
        ax3.legend(facecolor=BG,edgecolor="#1e3a52",labelcolor="#94a3b8",fontsize=8)
        ax3.grid(True,alpha=0.3); fig3.tight_layout(); st.pyplot(fig3); plt.close()
    st.markdown('<div class="sec" style="margin-top:1.5rem">Policy Recommendations</div>',unsafe_allow_html=True)
    p1,p2,p3=st.columns(3)
    for col,icon,title,text in [
        (p1,"\U0001f3e6","Expand Banking Infrastructure",
         "Prioritise rural branch expansion in low branch-density states: Bihar (0.06), UP (0.08), Jharkhand (0.08). Business Correspondent networks can substitute for full branches."),
        (p2,"\U0001f3e0","Strengthen Women's Autonomy",
         "Integrate financial literacy into SHG programmes to link decision-making empowerment directly with account opening and active usage by rural women."),
        (p3,"\U0001f467","Reduce Child Marriage",
         "Target high child-marriage states (West Bengal 48%, Tripura 42%, Bihar 43%) through conditional cash transfers tied to girls staying enrolled in school."),
    ]:
        with col:
            st.markdown(f'<div style="background:#0d2137;border:1px solid #1e3a52;border-radius:12px;padding:1.1rem 1.2rem">'
                        f'<div style="font-size:1.4rem;margin-bottom:.5rem">{icon}</div>'
                        f'<div style="color:#e2e8f0;font-weight:600;font-size:.88rem;margin-bottom:.5rem">{title}</div>'
                        f'<div style="color:#94a3b8;font-size:.8rem;line-height:1.6">{text}</div>'
                        f'</div>',unsafe_allow_html=True)
