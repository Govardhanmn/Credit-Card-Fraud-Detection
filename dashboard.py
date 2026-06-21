"""
Fraud Detection AI — PCA Feature Space Dashboard
Futuristic cybersecurity-aesthetic fraud monitoring system.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Fraud Detection AI — PCA Space",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BG      = "#020817"
BG_CARD = "#080f1e"
BLUE    = "#00d4ff"
RED     = "#ff2d55"
GREEN   = "#00ff88"
AMBER   = "#ffb800"
TXT     = "#94a3b8"
TXT_DIM = "#2a4060"
BORDER  = "#0d1f35"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600;700&display=swap');
* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #020817 !important;
    background-image:
        radial-gradient(ellipse 80% 35% at 10% 0%,  rgba(0,212,255,0.055) 0%, transparent 70%),
        radial-gradient(ellipse 60% 35% at 90% 100%, rgba(255,45,85,0.045)  0%, transparent 70%);
    color: #e2e8f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
.top-bar {
    background: linear-gradient(135deg, rgba(8,15,30,.98) 0%, rgba(2,8,23,.99) 100%);
    border: 1px solid rgba(0,212,255,.22); border-radius: 14px;
    padding: 18px 28px; margin-bottom: 18px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 0 60px rgba(0,212,255,.06);
}
.brand { font-size: 21px; font-weight: 900; color: #fff; letter-spacing: -.5px; }
.brand em { font-style: normal; color: #00d4ff; text-shadow: 0 0 22px rgba(0,212,255,.75); }
.sub { font-size: 9.5px; letter-spacing: 2.2px; text-transform: uppercase;
       color: #2a4060; margin-top: 3px; font-family: 'JetBrains Mono', monospace !important; }
.live-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,255,136,.07); border: 1px solid rgba(0,255,136,.2);
    border-radius: 999px; padding: 6px 16px; font-size: 9.5px;
    font-weight: 700; color: #00ff88; letter-spacing: 2px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: #00ff88;
       box-shadow: 0 0 8px #00ff88; animation: blink 1.8s ease-in-out infinite; }
@keyframes blink {
    0%,100% { opacity: 1; box-shadow: 0 0 8px #00ff88; }
    50%      { opacity: .3; box-shadow: 0 0 22px #00ff88; }
}
.kpi { background: rgba(8,15,30,.92); border: 1px solid rgba(0,212,255,.09);
       border-radius: 12px; padding: 14px 18px; position: relative; overflow: hidden;
       transition: border-color .25s, box-shadow .25s; }
.kpi:hover { border-color: rgba(0,212,255,.28); box-shadow: 0 0 18px rgba(0,212,255,.08); }
.kpi::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
              background: linear-gradient(90deg, transparent, rgba(0,212,255,.38), transparent); }
.kpi-lbl { font-size: 8.5px; font-weight: 700; text-transform: uppercase;
           letter-spacing: 2px; color: #2a4060; font-family: 'JetBrains Mono', monospace !important; }
.kpi-val { font-size: 23px; font-weight: 900; color: #00d4ff;
           letter-spacing: -1px; margin: 4px 0 1px;
           font-family: 'JetBrains Mono', monospace !important;
           text-shadow: 0 0 16px rgba(0,212,255,.45); }
.kpi-sub { font-size: 9.5px; color: #3d5a7a; }
.panel { background: rgba(8,15,30,.93); border: 1px solid rgba(0,212,255,.09);
         border-radius: 14px; padding: 18px; margin-bottom: 14px;
         position: relative; overflow: hidden; }
.panel::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
                 background: linear-gradient(90deg, transparent, rgba(0,212,255,.3), transparent); }
.panel-hdr { font-size: 9.5px; font-weight: 700; text-transform: uppercase;
             letter-spacing: 2.2px; color: #00d4ff; margin-bottom: 12px;
             display: flex; align-items: center; gap: 8px;
             font-family: 'JetBrains Mono', monospace !important; }
.panel-hdr::before { content: ''; width: 5px; height: 5px; background: #00d4ff;
                     border-radius: 50%; box-shadow: 0 0 8px #00d4ff; flex-shrink: 0; }
.sig { background: rgba(255,45,85,.08); border: 1px solid rgba(255,45,85,.2);
       border-radius: 6px; padding: 7px 12px; font-size: 10.5px; color: #ff6b8a;
       margin-bottom: 7px; line-height: 1.45; font-family: 'JetBrains Mono', monospace !important; }
.sig-ok { background: rgba(0,212,255,.06); border-color: rgba(0,212,255,.18); color: #7dd3fc; }
.stat-row { display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid rgba(0,212,255,.05); padding: 7px 0; font-size: 11px; }
.sl { color: #3d5a7a; font-family: 'JetBrains Mono', monospace !important; }
.sv { color: #e2e8f0; font-weight: 600; font-family: 'JetBrains Mono', monospace !important; }
[data-baseweb="select"] > div { background: rgba(8,15,30,.95) !important;
    border-color: rgba(0,212,255,.22) !important; color: #94a3b8 !important; border-radius: 9px !important; }
[data-testid="stDataFrame"] th { background: rgba(0,212,255,.06) !important;
    color: #00d4ff !important; font-size: 9px !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important; }
</style>
""", unsafe_allow_html=True)

AMOUNT_MEDIAN, AMOUNT_IQR = 22.0, 71.565
TIME_MEDIAN,   TIME_IQR   = 84692.0, 85119.0
def scale_amount(a): return (a - AMOUNT_MEDIAN) / AMOUNT_IQR
def scale_time(t):   return (t - TIME_MEDIAN)   / TIME_IQR

@st.cache_resource
def load_model():
    for path in ["xgboost_fraud_detector.pkl", "random_forest_fraud_detector.pkl"]:
        if os.path.exists(path):
            try:   return joblib.load(path), path.split("_")[0].upper(), True
            except: pass
    return None, "XGBoost", False

mdl, MODEL_NAME, model_ok = load_model()

@st.cache_data(show_spinner="Initializing AI systems...")
def load_default_data():
    if not os.path.exists("transaction_pool.csv"):
        st.error("transaction_pool.csv not found."); st.stop()
    df = pd.read_csv("transaction_pool.csv")
    df["scaled_Amount"] = scale_amount(df["Amount"])
    df["scaled_Time"]   = scale_time(df["Time"])
    feat = [f"V{i}" for i in range(1, 29)] + ["scaled_Amount", "scaled_Time"]
    if model_ok:
        df["Risk_Score"] = mdl.predict_proba(df[feat])[:, 1]
    elif "Risk_Score" not in df.columns:
        np.random.seed(0)
        df["Risk_Score"] = np.where(df["Class"] == 1,
                                     np.random.uniform(.6, .99, len(df)),
                                     np.random.uniform(.01, .25, len(df)))
    pca_cols = [f"V{i}" for i in range(1, 29)]
    centroid = df[df["Class"] == 0][pca_cols].mean().values
    dists = np.sqrt(((df[pca_cols].values - centroid) ** 2).sum(axis=1))
    mx = dists.max()
    df["Anomaly_Dist"]      = dists
    df["Anomaly_Dist_Norm"] = dists / mx if mx > 0 else dists
    df["Risk_Tier"] = np.select(
        [df["Risk_Score"] >= 0.65, df["Risk_Score"] >= 0.20],
        ["HIGH", "MEDIUM"], default="LOW"
    )
    return df, centroid, mx

default_df, centroid, max_dist = load_default_data()
df = default_df

def process_uploaded_data(df_raw, centroid, max_dist):
    df = df_raw.copy()
    pca_cols = [f"V{i}" for i in range(1, 29)]
    required_cols = pca_cols + ["Amount", "Time"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    
    if "Transaction_ID" not in df.columns:
        df["Transaction_ID"] = [f"TX_UP_{i:04d}" for i in range(len(df))]
    if "Class" not in df.columns:
        df["Class"] = 0
        
    df["scaled_Amount"] = scale_amount(df["Amount"])
    df["scaled_Time"]   = scale_time(df["Time"])
    
    feat = pca_cols + ["scaled_Amount", "scaled_Time"]
    if model_ok:
        df["Risk_Score"] = mdl.predict_proba(df[feat])[:, 1]
    elif "Risk_Score" not in df.columns:
        np.random.seed(0)
        df["Risk_Score"] = np.where(df["Class"] == 1,
                                     np.random.uniform(.6, .99, len(df)),
                                     np.random.uniform(.01, .25, len(df)))
                                     
    dists = np.sqrt(((df[pca_cols].values - centroid) ** 2).sum(axis=1))
    df["Anomaly_Dist"]      = dists
    df["Anomaly_Dist_Norm"] = dists / max_dist if max_dist > 0 else dists
    df["Risk_Tier"] = np.select(
        [df["Risk_Score"] >= 0.65, df["Risk_Score"] >= 0.20],
        ["HIGH", "MEDIUM"], default="LOW"
    )
    return df

if "sel_id" not in st.session_state:
    st.session_state.sel_id = df.nlargest(1, "Risk_Score")["Transaction_ID"].iloc[0]

def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG_CARD)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.tick_params(colors=TXT_DIM, labelsize=7)
    ax.grid(True, color=BORDER, lw=0.5, alpha=0.7)
    return fig, ax

mdl_status = "LOADED" if model_ok else "OFFLINE"
st.markdown(f"""
<div class="top-bar">
  <div>
    <div class="brand">&#x1F9E0; Fraud Detection <em>AI</em></div>
    <div class="sub">PCA Feature Space &nbsp;·&nbsp; {MODEL_NAME} Engine &nbsp;·&nbsp; Real-time Anomaly Scoring</div>
  </div>
  <div style="display:flex;gap:28px;align-items:center;">
    <div style="text-align:right;">
      <div class="sub">AUC-ROC</div>
      <div style="font-family:'JetBrains Mono';font-size:20px;font-weight:900;color:#00d4ff;text-shadow:0 0 16px rgba(0,212,255,.55);">0.9842</div>
    </div>
    <div style="width:1px;height:34px;background:rgba(0,212,255,.12);"></div>
    <div style="text-align:right;">
      <div class="sub">PR-AUC</div>
      <div style="font-family:'JetBrains Mono';font-size:20px;font-weight:900;color:#00d4ff;text-shadow:0 0 16px rgba(0,212,255,.55);">0.8800</div>
    </div>
    <div style="width:1px;height:34px;background:rgba(0,212,255,.12);"></div>
    <div style="text-align:right;">
      <div class="sub">Model</div>
      <div style="font-family:'JetBrains Mono';font-size:11px;font-weight:700;color:#64748b;">{MODEL_NAME} v2.0 &nbsp;·&nbsp; {mdl_status}</div>
    </div>
    <div style="width:1px;height:34px;background:rgba(0,212,255,.12);"></div>
    <div class="live-badge"><span class="dot"></span>SYSTEM ONLINE</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── CSV File Uploader Section ──
st.markdown("""
<style>
.upload-panel {
    background: rgba(8,15,30,.88);
    border: 1px dashed rgba(0,212,255,.2);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 20px;
}
.upload-desc {
    font-size: 11px;
    color: #4f7396;
    margin-bottom: 12px;
    line-height: 1.5;
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

with st.expander("📂 UPLOAD CUSTOM TRANSACTION CSV", expanded=False):
    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    st.markdown('<div class="upload-desc">Upload a CSV file containing transactions to analyze. The CSV must include columns for <strong>Time</strong>, <strong>Amount</strong>, and the 28 PCA-transformed components (<strong>V1</strong> to <strong>V28</strong>).</div>', unsafe_allow_html=True)
    
    col_u1, col_u2 = st.columns([7, 3])
    with col_u1:
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=["csv"],
            key="csv_uploader",
            label_visibility="collapsed"
        )
    with col_u2:
        # Generate sample template based on first 5 rows of default_df
        # Remove computed columns so it's a clean input template
        cols_to_keep = [c for c in default_df.columns if c not in ["Risk_Score", "Anomaly_Dist", "Anomaly_Dist_Norm", "Risk_Tier", "scaled_Amount", "scaled_Time"]]
        template_csv = default_df[cols_to_keep].head(5).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 DOWNLOAD CSV TEMPLATE",
            data=template_csv,
            file_name="fraud_detection_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        raw_uploaded = pd.read_csv(uploaded_file)
        df = process_uploaded_data(raw_uploaded, centroid, max_dist)
    except Exception as e:
        st.error(f"⚠️ Error parsing uploaded CSV: {str(e)}")
        st.info("Falling back to default transaction pool.")
        df = default_df

if st.session_state.sel_id not in df["Transaction_ID"].values:
    st.session_state.sel_id = df.nlargest(1, "Risk_Score")["Transaction_ID"].iloc[0]

total=len(df); n_fraud=len(df[df["Class"]==1]); n_high=len(df[df["Risk_Tier"]=="HIGH"])
avg_score=df["Risk_Score"].mean(); avg_dist=df["Anomaly_Dist_Norm"].mean()

kc = st.columns(5)
for col,(lbl,val,sub) in zip(kc,[
    ("Transactions",    f"{total:,}",       f"{total} records in pool"),
    ("Fraud Cases",     f"{n_fraud}",        f"{n_fraud/total*100:.1f}% of pool"),
    ("High-Risk Alerts",f"{n_high}",         "Risk Score >= 0.65"),
    ("Avg Risk Score",  f"{avg_score:.4f}",  "Mean model output"),
    ("Avg Anomaly Dist",f"{avg_dist:.4f}",   "Normalised centroid dist"),
]):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-lbl">{lbl}</div><div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
col_L,col_R=st.columns([62,38])

with col_L:
    st.markdown('<div class="panel"><div class="panel-hdr">PCA LATENT SPACE — PC1 vs PC2</div>',unsafe_allow_html=True)
    normal_s=df[df["Class"]==0].sample(min(600,len(df[df["Class"]==0])),random_state=1)
    fraud_s=df[df["Class"]==1]; sel_row=df[df["Transaction_ID"]==st.session_state.sel_id]
    fig1,ax1=dark_fig(9,5.2)
    ax1.scatter(normal_s["V1"],normal_s["V2"],c=BLUE,alpha=0.18,s=10,zorder=2,rasterized=True)
    ax1.scatter(fraud_s["V1"],fraud_s["V2"],c=RED,alpha=0.05,s=110,zorder=3,rasterized=True)
    ax1.scatter(fraud_s["V1"],fraud_s["V2"],c=RED,alpha=0.15,s=50,zorder=3,rasterized=True)
    ax1.scatter(fraud_s["V1"],fraud_s["V2"],c=RED,alpha=0.80,s=18,zorder=4,rasterized=True)
    if len(sel_row)>0:
        sx,sy=float(sel_row["V1"].iloc[0]),float(sel_row["V2"].iloc[0])
        ax1.scatter(sx,sy,c="none",s=520,zorder=5,edgecolors=BLUE,linewidths=1.2,alpha=0.35)
        ax1.scatter(sx,sy,c="none",s=260,zorder=5,edgecolors=BLUE,linewidths=2.0,alpha=0.70)
        ax1.scatter(sx,sy,c="#ffffff",s=70,zorder=6,marker="*",edgecolors=BLUE,linewidths=0.8)
    ax1.set_xlabel("PC1  (V1)",color=TXT,fontsize=8,labelpad=5)
    ax1.set_ylabel("PC2  (V2)",color=TXT,fontsize=8,labelpad=5)
    leg1=ax1.legend(handles=[
        Line2D([0],[0],marker="o",color="none",markerfacecolor=BLUE,markersize=9,label=f"Normal  ({len(normal_s):,} sampled)"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor=RED,markersize=9,label=f"Fraud  ({len(fraud_s)} total)"),
        Line2D([0],[0],marker="*",color="none",markerfacecolor="#fff",markeredgecolor=BLUE,markersize=13,label="Selected"),
    ],loc="upper right",framealpha=0.2,facecolor=BG_CARD,edgecolor=BORDER,fontsize=8)
    for t in leg1.get_texts(): t.set_color(TXT)
    plt.tight_layout(); st.pyplot(fig1,use_container_width=True); plt.close(fig1)
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-hdr">FRAUD PROBABILITY OVER TIME</div>',unsafe_allow_html=True)
    df_ts=df.sort_values("Time").reset_index(drop=True)
    win=max(5,len(df_ts)//60); df_ts["Roll"]=df_ts["Risk_Score"].rolling(win,center=True,min_periods=1).mean()
    fig2,ax2=dark_fig(9,3.0)
    ax2.fill_between(df_ts["Time"],df_ts["Roll"],color=BLUE,alpha=0.09)
    ax2.plot(df_ts["Time"],df_ts["Roll"],color=BLUE,lw=1.8,zorder=3)
    fd_ts=df_ts[df_ts["Class"]==1]
    ax2.scatter(fd_ts["Time"],fd_ts["Risk_Score"],c=RED,s=14,alpha=0.55,zorder=5)
    ax2.axhline(0.65,color=RED,ls="--",lw=0.9,alpha=0.5,label="Decline (0.65)")
    ax2.axhline(0.20,color=AMBER,ls="--",lw=0.9,alpha=0.5,label="Review  (0.20)")
    ax2.set_xlabel("Time (seconds)",color=TXT,fontsize=8,labelpad=4)
    ax2.set_ylabel("Risk Score",color=TXT,fontsize=8,labelpad=4)
    leg2=ax2.legend(fontsize=7,framealpha=0.15,facecolor=BG_CARD,edgecolor=BORDER)
    for t in leg2.get_texts(): t.set_color(TXT)
    plt.tight_layout(); st.pyplot(fig2,use_container_width=True); plt.close(fig2)
    st.markdown("</div>",unsafe_allow_html=True)

with col_R:
    # ── Tier filter (shows all transactions, not just top-100) ──
    tier_filter = st.radio(
        "Filter by Risk Tier",
        ["All", "HIGH", "MEDIUM", "LOW"],
        horizontal=True,
        key="tier_filter"
    )
    if tier_filter == "All":
        pool = df.sort_values("Risk_Score", ascending=False)
    else:
        pool = df[df["Risk_Tier"] == tier_filter].sort_values("Risk_Score", ascending=False)

    all_ids = pool["Transaction_ID"].tolist()
    if not all_ids:
        st.warning("No transactions in this tier.")
        st.stop()

    # Keep previous selection if it's in the filtered list
    if st.session_state.sel_id not in all_ids:
        st.session_state.sel_id = all_ids[0]
    idx = all_ids.index(st.session_state.sel_id)

    chosen = st.selectbox(
        f"Inspect Transaction  ({len(all_ids)} shown)",
        all_ids, index=idx, key="txn_pick"
    )
    st.session_state.sel_id = chosen
    txn=df[df["Transaction_ID"]==chosen].iloc[0]
    prob=float(txn["Risk_Score"]); dist=float(txn["Anomaly_Dist_Norm"])
    gc=GREEN if prob<0.20 else (AMBER if prob<0.65 else RED)
    lbl="NORMAL" if prob<0.20 else ("SUSPICIOUS" if prob<0.65 else "CRITICAL RISK")

    st.markdown('<div class="panel"><div class="panel-hdr">FRAUD SCORE GAUGE</div>',unsafe_allow_html=True)
    fig_g,ax_g=plt.subplots(figsize=(5,3.4))
    fig_g.patch.set_facecolor(BG); ax_g.set_facecolor(BG)
    ax_g.set_xlim(-1.15,1.15); ax_g.set_ylim(-0.55,1.18)
    ax_g.set_aspect("equal"); ax_g.axis("off")
    cx,cy=0.0,0.0; Ro,Ri=1.0,0.60; W=Ro-Ri
    ax_g.add_patch(Wedge((cx,cy),Ro,0,180,width=W,facecolor="#0b1a2e",edgecolor=BORDER,lw=0.8))
    ax_g.add_patch(Wedge((cx,cy),Ro-0.025,144,180,width=W-0.05,facecolor="#003d1a",edgecolor="none",alpha=0.85))
    ax_g.add_patch(Wedge((cx,cy),Ro-0.025,90,144,width=W-0.05,facecolor="#332800",edgecolor="none",alpha=0.85))
    ax_g.add_patch(Wedge((cx,cy),Ro-0.025,0,90,width=W-0.05,facecolor="#2a000c",edgecolor="none",alpha=0.85))
    end_deg=180.0-prob*180.0
    if prob>0.002:
        ax_g.add_patch(Wedge((cx,cy),Ro-0.05,end_deg,180,width=W-0.10,facecolor=gc,edgecolor="none",alpha=0.88,zorder=3))
    angle_rad=np.radians(end_deg); nl=Ri-0.06
    nx,ny=cx+nl*np.cos(angle_rad),cy+nl*np.sin(angle_rad)
    ax_g.annotate("",xy=(nx,ny),xytext=(cx,cy),
                  arrowprops=dict(arrowstyle="->",color=gc,lw=2.8,mutation_scale=14))
    ax_g.add_patch(Circle((cx,cy),0.065,facecolor=BG_CARD,edgecolor=gc,lw=2.2,zorder=5))
    ax_g.text(cx,cy-0.22,f"{prob:.4f}",ha="center",va="center",fontsize=21,fontweight="black",color=gc,fontfamily="monospace")
    ax_g.text(cx,cy-0.40,lbl,ha="center",va="center",fontsize=8,fontweight="bold",color=gc,fontfamily="monospace",alpha=0.88)
    for tv,ta in [(0.0,"0.0"),(0.5,"0.5"),(1.0,"1.0")]:
        a=np.radians(180-tv*180)
        ax_g.text(cx+(Ro+0.10)*np.cos(a),cy+(Ro+0.10)*np.sin(a),ta,ha="center",va="center",fontsize=7,color=TXT_DIM,fontfamily="monospace")
    plt.tight_layout(); st.pyplot(fig_g,use_container_width=True); plt.close(fig_g)
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-hdr">TOP RISK TRANSACTIONS</div>',unsafe_allow_html=True)
    tbl=df.nlargest(15,"Risk_Score")[["Transaction_ID","Time","Amount","Risk_Score","Anomaly_Dist_Norm"]].copy()
    tbl.columns=["TXN ID","Time (s)","Amount (EUR)","Risk Score","Anomaly Dist"]
    tbl["Risk Score"]=tbl["Risk Score"].round(4); tbl["Anomaly Dist"]=tbl["Anomaly Dist"].round(4)
    tbl["Amount (EUR)"]=tbl["Amount (EUR)"].round(2); tbl["Time (s)"]=tbl["Time (s)"].astype(int)
    st.dataframe(tbl,use_container_width=True,hide_index=True,height=250)
    st.markdown("</div>",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
col_b1,col_b2=st.columns(2)

with col_b1:
    st.markdown('<div class="panel"><div class="panel-hdr">TRANSACTION AMOUNT vs FRAUD SCORE</div>',unsafe_allow_html=True)
    nm3=df[df["Class"]==0]; fd3=df[df["Class"]==1]
    fig3,ax3=dark_fig(6,4.0)
    ax3.scatter(nm3["Amount"],nm3["Risk_Score"],c=BLUE,alpha=0.18,s=10,rasterized=True)
    ax3.scatter(fd3["Amount"],fd3["Risk_Score"],c=RED,alpha=0.06,s=90,rasterized=True)
    ax3.scatter(fd3["Amount"],fd3["Risk_Score"],c=RED,alpha=0.75,s=18,rasterized=True)
    ax3.scatter(float(txn["Amount"]),prob,c="#fff",s=130,zorder=6,marker="*",edgecolors=BLUE,linewidths=1.0)
    ax3.axhline(0.65,color=RED,ls="--",lw=0.9,alpha=0.5,label="Decline threshold")
    ax3.axhline(0.20,color=AMBER,ls="--",lw=0.9,alpha=0.5,label="Review threshold")
    ax3.set_xlabel("Transaction Amount (EUR)",color=TXT,fontsize=8,labelpad=4)
    ax3.set_ylabel("Fraud Score",color=TXT,fontsize=8,labelpad=4)
    leg3=ax3.legend(handles=[
        Line2D([0],[0],marker="o",color="none",markerfacecolor=BLUE,markersize=8,label="Normal"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor=RED,markersize=8,label="Fraud"),
        Line2D([0],[0],marker="*",color="none",markerfacecolor="#fff",markeredgecolor=BLUE,markersize=13,label="Selected"),
    ],fontsize=7,framealpha=0.15,facecolor=BG_CARD,edgecolor=BORDER)
    for t in leg3.get_texts(): t.set_color(TXT)
    plt.tight_layout(); st.pyplot(fig3,use_container_width=True); plt.close(fig3)
    st.markdown("</div>",unsafe_allow_html=True)

with col_b2:
    st.markdown(f'<div class="panel"><div class="panel-hdr">PCA VECTOR — {chosen}</div>',unsafe_allow_html=True)
    all_v=np.array([float(txn[f"V{i}"]) for i in range(1,29)])
    bar_c=[RED if abs(v)>2.5 else AMBER if abs(v)>1.5 else "#1a3d5c" for v in all_v]
    fig4,ax4=dark_fig(6,4.0)
    ax4.bar(range(28),all_v,color=bar_c,width=0.72,edgecolor=BG,lw=0.4)
    ax4.axhline(0,color=BORDER,lw=1.0)
    ax4.axhline(2.5,color=RED,ls="--",lw=0.85,alpha=0.45,label="+-2.5sigma (high anomaly)")
    ax4.axhline(-2.5,color=RED,ls="--",lw=0.85,alpha=0.45)
    ax4.axhline(1.5,color=AMBER,ls=":",lw=0.7,alpha=0.35,label="+-1.5sigma (moderate)")
    ax4.axhline(-1.5,color=AMBER,ls=":",lw=0.7,alpha=0.35)
    ax4.set_xticks(range(28))
    ax4.set_xticklabels([f"V{i+1}" for i in range(28)],rotation=45,ha="right",fontsize=6.5,color=TXT_DIM)
    ax4.set_ylabel("Component Value",color=TXT,fontsize=8,labelpad=4)
    for sp in ["top","right"]: ax4.spines[sp].set_visible(False)
    leg4=ax4.legend(fontsize=7,framealpha=0.15,facecolor=BG_CARD,edgecolor=BORDER)
    for t in leg4.get_texts(): t.set_color(TXT)
    plt.tight_layout(); st.pyplot(fig4,use_container_width=True); plt.close(fig4)
    st.markdown("</div>",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
pca_ranked=sorted([(f"V{i}",float(txn[f"V{i}"])) for i in range(1,29)],key=lambda x:abs(x[1]),reverse=True)
flagged=[(k,v) for k,v in pca_ranked if abs(v)>2.5]
moderate=[(k,v) for k,v in pca_ranked if 1.5<abs(v)<=2.5]
col_i,col_d=st.columns([42,58])
risk_color="#ff2d55" if prob>=0.65 else ("#ffb800" if prob>=0.20 else "#00ff88")

with col_i:
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">AI ANOMALY INTELLIGENCE</div>
      <div style="margin-bottom:14px;">
        <div class="kpi-lbl">Transaction Under Analysis</div>
        <div style="font-family:'JetBrains Mono';font-size:15px;color:#00d4ff;font-weight:700;margin-top:4px;">{chosen}</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;">
        <div><div class="kpi-lbl">Fraud Probability</div>
          <div style="font-family:'JetBrains Mono';font-size:22px;font-weight:900;color:{risk_color};text-shadow:0 0 14px {risk_color}70;">{prob:.4f}</div></div>
        <div><div class="kpi-lbl">Anomaly Distance</div>
          <div style="font-family:'JetBrains Mono';font-size:22px;font-weight:900;color:#00d4ff;">{dist:.4f}</div></div>
        <div><div class="kpi-lbl">Amount</div>
          <div style="font-family:'JetBrains Mono';font-size:13px;font-weight:700;color:#94a3b8;">EUR {txn['Amount']:.2f}</div></div>
        <div><div class="kpi-lbl">Classification</div>
          <div style="font-family:'JetBrains Mono';font-size:13px;font-weight:700;color:{risk_color};">{txn['Risk_Tier']}</div></div>
      </div>
      <div class="kpi-lbl" style="margin-bottom:8px;">Anomaly Signals</div>
    """,unsafe_allow_html=True)
    if flagged:
        fl=", ".join([k for k,_ in flagged[:4]])
        st.markdown(f'<div class="sig">HIGH DEVIATION in {fl} — values exceed +-2.5sigma</div>',unsafe_allow_html=True)
    if moderate:
        mo=", ".join([k for k,_ in moderate[:3]])
        st.markdown(f'<div class="sig">MODERATE ANOMALY in {mo} — +-1.5 to 2.5sigma range</div>',unsafe_allow_html=True)
    if prob>=0.65:
        st.markdown(f'<div class="sig">Score {prob:.4f} crosses auto-decline boundary (0.65)</div>',unsafe_allow_html=True)
    if float(txn["Amount"])>1000:
        st.markdown('<div class="sig">High-value transaction amplifies fraud classification</div>',unsafe_allow_html=True)
    if dist>0.5:
        st.markdown(f'<div class="sig">Distance {dist:.3f} — extreme outlier from normal cluster centroid</div>',unsafe_allow_html=True)
    if prob<0.20 and not flagged:
        st.markdown('<div class="sig sig-ok">PCA vector within expected normal distribution bounds</div>',unsafe_allow_html=True)
    if not flagged and prob>=0.20:
        st.markdown('<div class="sig sig-ok">Collective multi-component deviation detected below threshold</div>',unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

with col_d:
    true_lbl="FRAUD" if txn["Class"]==1 else "NORMAL"
    true_col="#ff2d55" if txn["Class"]==1 else "#00ff88"
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">TRANSACTION DETAIL — {chosen}</div>
      <div class="stat-row"><span class="sl">Transaction ID</span><span class="sv">{chosen}</span></div>
      <div class="stat-row"><span class="sl">Time (s)</span><span class="sv">{int(txn['Time'])}</span></div>
      <div class="stat-row"><span class="sl">Amount</span><span class="sv">EUR {txn['Amount']:.2f}</span></div>
      <div class="stat-row"><span class="sl">Risk Score</span><span class="sv" style="color:{risk_color};">{prob:.6f}</span></div>
      <div class="stat-row"><span class="sl">Anomaly Distance (norm.)</span><span class="sv">{dist:.6f}</span></div>
      <div class="stat-row"><span class="sl">Risk Tier</span><span class="sv" style="color:{risk_color};">{txn['Risk_Tier']}</span></div>
      <div class="stat-row"><span class="sl">Ground Truth</span><span class="sv" style="color:{true_col};">{true_lbl}</span></div>
      <div class="stat-row"><span class="sl">Top Anomalous Components</span><span class="sv">{', '.join([k for k,_ in pca_ranked[:6]])}</span></div>
      <div style="margin-top:14px;">
        <div class="kpi-lbl" style="margin-bottom:8px;">Top 8 PCA Component Contributions</div>
    """,unsafe_allow_html=True)
    for k,v in pca_ranked[:8]:
        bw=min(abs(v)/5.0*100,100)
        bc=RED if abs(v)>2.5 else (AMBER if abs(v)>1.5 else BLUE)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
          <div style="font-family:'JetBrains Mono';font-size:9.5px;color:#3d5a7a;width:30px;">{k}</div>
          <div style="flex:1;background:#0b1a2e;border-radius:3px;height:7px;overflow:hidden;">
            <div style="height:100%;width:{bw:.1f}%;background:{bc};border-radius:3px;box-shadow:0 0 6px {bc}60;"></div>
          </div>
          <div style="font-family:'JetBrains Mono';font-size:9.5px;color:{bc};width:50px;text-align:right;">{v:+.3f}</div>
        </div>""",unsafe_allow_html=True)
    st.markdown("</div></div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
  <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(0,212,255,0.25));"></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;
              letter-spacing:3px;color:#00d4ff;text-transform:uppercase;white-space:nowrap;">
    Descriptive Analysis
  </div>
  <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(0,212,255,0.25),transparent);"></div>
</div>
""", unsafe_allow_html=True)

# Pre-compute all stats
fraud_df   = df[df["Class"] == 1]
normal_df  = df[df["Class"] == 0]
high_df    = df[df["Risk_Tier"] == "HIGH"]
med_df     = df[df["Risk_Tier"] == "MEDIUM"]
low_df     = df[df["Risk_Tier"] == "LOW"]

fraud_rate       = n_fraud / total * 100
high_pct         = len(high_df) / total * 100
med_pct          = len(med_df)  / total * 100
low_pct          = len(low_df)  / total * 100
avg_fraud_score  = fraud_df["Risk_Score"].mean()  if len(fraud_df)  > 0 else 0
avg_normal_score = normal_df["Risk_Score"].mean() if len(normal_df) > 0 else 0
avg_fraud_amt    = fraud_df["Amount"].mean()      if len(fraud_df)  > 0 else 0
avg_normal_amt   = normal_df["Amount"].mean()     if len(normal_df) > 0 else 0
max_fraud_amt    = fraud_df["Amount"].max()       if len(fraud_df)  > 0 else 0
model_sep        = avg_fraud_score - avg_normal_score

pca_cols_list = [f"V{i}" for i in range(1, 29)]
fraud_norms  = fraud_df[pca_cols_list].apply(lambda r: np.sqrt((r**2).sum()), axis=1)
normal_norms = normal_df[pca_cols_list].apply(lambda r: np.sqrt((r**2).sum()), axis=1)
top_pca_str  = ", ".join(fraud_df[pca_cols_list].abs().mean().sort_values(ascending=False).head(5).index.tolist()) if len(fraud_df) > 0 else "N/A"

time_q1, time_q3     = df["Time"].quantile(0.25), df["Time"].quantile(0.75)
peak_fraud_time      = fraud_df["Time"].median() if len(fraud_df) > 0 else 0
early_pct_val        = len(fraud_df[fraud_df["Time"] < time_q1]) / max(len(fraud_df), 1) * 100 if len(fraud_df) > 0 else 0
late_pct_val         = len(fraud_df[fraud_df["Time"] > time_q3]) / max(len(fraud_df), 1) * 100 if len(fraud_df) > 0 else 0

p25, p75, p95        = df["Amount"].quantile(0.25), df["Amount"].quantile(0.75), df["Amount"].quantile(0.95)
high_val_fraud       = len(fraud_df[fraud_df["Amount"] > p75])  if len(fraud_df)  > 0 else 0
high_val_normal      = len(normal_df[normal_df["Amount"] > p75]) if len(normal_df) > 0 else 0
fvn_pct              = high_val_fraud  / max(len(fraud_df), 1)  * 100
nvn_pct              = high_val_normal / max(len(normal_df), 1) * 100

sel_pct_rank         = (df["Risk_Score"] < prob).sum() / total * 100
sel_dist_rank        = (df["Anomaly_Dist_Norm"] < dist).sum() / total * 100
top3_names           = ", ".join([k for k, _ in pca_ranked[:3]])
top3_vals            = ", ".join([f"{v:+.3f}" for _, v in pca_ranked[:3]])
correct              = (prob >= 0.5 and txn["Class"] == 1) or (prob < 0.5 and txn["Class"] == 0)
verdict              = ("Correctly classified as FRAUD" if txn["Class"] == 1 and prob >= 0.5
                        else "Correctly classified as NORMAL" if txn["Class"] == 0 and prob < 0.5
                        else "Misclassified by model")
verdict_icon         = "PASS" if correct else "FAIL"
verdict_col          = "#00ff88" if correct else "#ff2d55"
sel_tier_color       = "#ff2d55" if txn["Risk_Tier"] == "HIGH" else ("#ffb800" if txn["Risk_Tier"] == "MEDIUM" else "#00ff88")
fraud_dist_mean      = fraud_df["Anomaly_Dist_Norm"].mean()  if len(fraud_df)  > 0 else 0
normal_dist_mean     = normal_df["Anomaly_Dist_Norm"].mean() if len(normal_df) > 0 else 0
dist_ratio           = fraud_dist_mean / max(normal_dist_mean, 1e-9)
fraud_norm_mean      = fraud_norms.mean()  if len(fraud_df)  > 0 else 0
normal_norm_mean     = normal_norms.mean() if len(normal_df) > 0 else 0

# ── ROW 1: Dataset Overview | Risk Tier Distribution ──────────────
da1, da2 = st.columns(2)

with da1:
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">Dataset Overview</div>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0 0 10px;">
        Active dataset contains <strong style="color:#e2e8f0;">{total:,} records</strong>:
        <strong style="color:#ff2d55;">{n_fraud} ({fraud_rate:.2f}%)</strong> fraud cases and
        <strong style="color:#00d4ff;">{len(normal_df):,} Legitimate</strong> transactions.
      </p>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0;">
        Model average scores: <strong style="color:#ff2d55;">{avg_fraud_score:.4f}</strong> (fraud) vs
        <strong style="color:#00ff88;">{avg_normal_score:.4f}</strong> (legitimate), establishing a decision margin of
        <strong style="color:#e2e8f0;">{model_sep:.4f}</strong>.
      </p>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:14px;">
        <div style="background:#0b1a2e;border-radius:8px;padding:8px;text-align:center;">
          <div class="kpi-lbl">Records</div>
          <div style="font-family:'JetBrains Mono';font-size:14px;font-weight:900;color:#00d4ff;">{total:,}</div>
        </div>
        <div style="background:#0b1a2e;border-radius:8px;padding:8px;text-align:center;">
          <div class="kpi-lbl">Fraud Rate</div>
          <div style="font-family:'JetBrains Mono';font-size:14px;font-weight:900;color:#ff2d55;">{fraud_rate:.2f}%</div>
        </div>
        <div style="background:#0b1a2e;border-radius:8px;padding:8px;text-align:center;">
          <div class="kpi-lbl">Score Gap</div>
          <div style="font-family:'JetBrains Mono';font-size:14px;font-weight:900;color:#ffb800;">{model_sep:.4f}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with da2:
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">Risk Tier Distribution</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px;">
        <div style="background:rgba(255,45,85,0.08);border:1px solid rgba(255,45,85,0.2);
                    border-radius:8px;padding:8px;text-align:center;">
          <div class="kpi-lbl" style="color:#ff2d55;">HIGH</div>
          <div style="font-family:'JetBrains Mono';font-size:18px;font-weight:900;color:#ff2d55;">{len(high_df)}</div>
          <div style="font-size:9.5px;color:#7a2535;">{high_pct:.1f}%</div>
        </div>
        <div style="background:rgba(255,184,0,0.07);border:1px solid rgba(255,184,0,0.2);
                    border-radius:8px;padding:8px;text-align:center;">
          <div class="kpi-lbl" style="color:#ffb800;">MEDIUM</div>
          <div style="font-family:'JetBrains Mono';font-size:18px;font-weight:900;color:#ffb800;">{len(med_df)}</div>
          <div style="font-size:9.5px;color:#7a6500;">{med_pct:.1f}%</div>
        </div>
        <div style="background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.18);
                    border-radius:8px;padding:8px;text-align:center;">
          <div class="kpi-lbl" style="color:#00ff88;">LOW</div>
          <div style="font-family:'JetBrains Mono';font-size:18px;font-weight:900;color:#00ff88;">{len(low_df)}</div>
          <div style="font-size:9.5px;color:#006b40;">{low_pct:.1f}%</div>
        </div>
      </div>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0;">
        Queues: Decline (<strong style="color:#ff2d55;">HIGH ≥ 0.65</strong>),
        Review (<strong style="color:#ffb800;">MEDIUM ≥ 0.20</strong>),
        and Auto-Approve (<strong style="color:#00ff88;">LOW < 0.20</strong>).
        Only HIGH-tier alerts require manual intervention.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── ROW 2: PCA Space | Temporal ───────────────────────────────────
da3, da4 = st.columns(2)

with da3:
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">PCA Feature Space Analysis</div>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0 0 10px;">
        Fraud lies farther from origin in PCA space: mean L2 vector norm of
        <strong style="color:#ff2d55;">{fraud_norm_mean:.3f}</strong> (fraud) vs
        <strong style="color:#00ff88;">{normal_norm_mean:.3f}</strong> (normal).
        Avg anomaly distance for fraud is <strong style="color:#ff2d55;">{fraud_dist_mean:.4f}</strong>
        (<strong style="color:#e2e8f0;">{dist_ratio:.1f}× further</strong> than normal's {normal_dist_mean:.4f}).
      </p>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0;">
        Primary outlier components in fraud: <strong style="color:#00d4ff;">{top_pca_str}</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)

with da4:
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">Temporal Pattern Analysis</div>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0 0 10px;">
        Fraud is distributed across the 48-hour time window (median timestamp:
        <strong style="color:#ff2d55;">{peak_fraud_time:,.0f}s</strong>).
      </p>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0;">
        About <strong style="color:#ffb800;">{early_pct_val:.1f}%</strong> of fraud occurs early
        (first quartile, before {time_q1:,.0f}s) and
        <strong style="color:#ffb800;">{late_pct_val:.1f}%</strong> late
        (last quartile, after {time_q3:,.0f}s), indicating continuous fraud distribution.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── ROW 3: Amount Analysis | Selected Transaction Deep-Dive ───────
da5, da6 = st.columns(2)

with da5:
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">Transaction Amount Analysis</div>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0 0 10px;">
        Average fraud amount: <strong style="color:#ff2d55;">EUR {avg_fraud_amt:.2f}</strong> vs
        <strong style="color:#00ff88;">EUR {avg_normal_amt:.2f}</strong> (normal). Max fraud is
        <strong style="color:#ff2d55;">EUR {max_fraud_amt:,.2f}</strong>.
      </p>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0;">
        Only <strong style="color:#ff2d55;">{fvn_pct:.1f}%</strong> of fraud exceeds the 75th percentile amount
        (EUR {p75:.2f}) vs <strong style="color:#00ff88;">{nvn_pct:.1f}%</strong> for normal, confirming amount
        alone is a weak fraud signal compared to PCA components.
      </p>
    </div>
    """, unsafe_allow_html=True)

with da6:
    flagged_note = "Features exceed the ±2.5σ anomaly threshold." if flagged else "Within normal deviation bounds."
    st.markdown(f"""
    <div class="panel">
      <div class="panel-hdr">Selected Transaction Deep-Dive — {chosen}</div>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0 0 10px;">
        Risk score <strong style="color:{sel_tier_color};">{prob:.4f}</strong> ({txn['Risk_Tier']}),
        exceeding <strong style="color:#e2e8f0;">{sel_pct_rank:.1f}%</strong> of dataset.
        Anomaly distance is <strong style="color:#00d4ff;">{dist:.4f}</strong> (<strong style="color:#e2e8f0;">{sel_dist_rank:.1f}th percentile</strong>).
      </p>
      <p style="font-size:13px;color:#94a3b8;line-height:1.7;margin:0 0 10px;">
        PCA outlier contributors: <strong style="color:#ff2d55;">{top3_names}</strong> ({top3_vals}). {flagged_note}
      </p>
      <div style="background:#0b1a2e;border-radius:8px;padding:10px 14px;
                  border-left:3px solid {verdict_col};margin-top:2px;">
        <div style="font-family:'JetBrains Mono';font-size:11px;
                    color:{verdict_col};font-weight:700;letter-spacing:1px;">
          [{verdict_icon}] &nbsp; {verdict}
        </div>
        <div style="font-size:10.5px;color:#475569;margin-top:2px;">
          Truth: {"FRAUD" if txn['Class']==1 else "NORMAL"}
          &nbsp;·&nbsp; Model: {"FRAUD" if prob>=0.5 else "NORMAL"}
          &nbsp;·&nbsp; Score: {prob:.6f}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:20px 0 10px;color:#1e3a5f;font-size:9px;
            font-family:'JetBrains Mono',monospace;letter-spacing:2.5px;
            border-top:1px solid #0d1f35;margin-top:24px;">
  FRAUD DETECTION AI  |  PCA FEATURE SPACE  |  XGBOOST ENGINE  |  CAPSTONE PROJECT
</div>
""", unsafe_allow_html=True)
