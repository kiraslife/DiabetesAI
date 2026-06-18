"""
app.py v2 — Premium redesigned Streamlit app with animated gauge,
radar chart, inline SHAP, and glassmorphism dark UI.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import streamlit as st
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI — Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: radial-gradient(ellipse at 20% 50%, #0d1b2a 0%, #080d13 60%, #0a0514 100%); }

/* Remove default padding */
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(30,58,95,0.8) 0%, rgba(10,5,25,0.9) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 24px;
    padding: 36px 40px;
    margin-bottom: 24px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 80px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero h1 { font-size:2.6rem; font-weight:900; color:#e8f4ff; margin:0 0 6px; letter-spacing:-1px; }
.hero h1 span { background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p  { color:#8bafd4; font-size:1.05rem; margin:0; }

/* Metric cards */
.metric-row { display:flex; gap:14px; margin-bottom:22px; }
.metric-card {
    flex:1;
    background: linear-gradient(145deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: border-color .2s;
}
.metric-card:hover { border-color: rgba(99,179,237,0.4); }
.metric-label { font-size:.75rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-size:1.9rem; font-weight:800; margin:4px 0 0; }
.metric-value.blue  { color:#60a5fa; }
.metric-value.green { color:#34d399; }
.metric-value.purple{ color:#a78bfa; }
.metric-value.rose  { color:#f472b6; }

/* Section headers */
.section-hdr {
    font-size:1rem; font-weight:700; color:#94a3b8;
    text-transform:uppercase; letter-spacing:.1em;
    margin:20px 0 12px;
    padding-bottom:8px;
    border-bottom:1px solid rgba(99,179,237,0.12);
}

/* Input panel */
.input-panel {
    background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(10,15,30,0.95));
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 20px;
    padding: 26px;
    backdrop-filter: blur(20px);
}

/* Sliders */
.stSlider > label  { color:#94a3b8 !important; font-size:.82rem !important; font-weight:500 !important; }
[data-testid="stSlider"] div[role="slider"] { background:#6366f1 !important; }

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 0 !important;
    width: 100% !important;
    letter-spacing: .02em !important;
    box-shadow: 0 4px 30px rgba(99,102,241,0.4) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(99,102,241,0.6) !important;
}

/* Result cards */
.result-box {
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin: 16px 0;
}
.result-low    { background: linear-gradient(135deg,rgba(6,78,59,.8),rgba(4,47,46,.9)); border:1px solid #10b981; }
.result-medium { background: linear-gradient(135deg,rgba(78,52,6,.8),rgba(47,31,4,.9)); border:1px solid #f59e0b; }
.result-high   { background: linear-gradient(135deg,rgba(78,6,18,.8),rgba(47,4,11,.9)); border:1px solid #ef4444; }
.result-pct  { font-size:3.5rem; font-weight:900; margin:8px 0; }
.result-tier { font-size:1.1rem; font-weight:700; letter-spacing:.05em; }
.result-note { font-size:.85rem; color:rgba(255,255,255,.55); margin-top:8px; }

/* Driver cards */
.driver {
    display:flex; align-items:center; gap:14px;
    background:rgba(30,41,59,.7);
    border:1px solid rgba(99,179,237,.1);
    border-radius:12px;
    padding:14px 16px;
    margin:8px 0;
    transition: border-color .2s;
}
.driver:hover { border-color:rgba(99,179,237,.3); }
.driver-icon { font-size:1.5rem; min-width:32px; text-align:center; }
.driver-name { font-size:.9rem; font-weight:600; color:#e2e8f0; }
.driver-desc { font-size:.78rem; color:#64748b; margin-top:2px; }
.driver-bar-wrap { flex:1; background:rgba(255,255,255,.06); border-radius:999px; height:6px; overflow:hidden; }
.driver-bar { height:100%; border-radius:999px; }

/* Disclaimer */
.disclaimer {
    background:rgba(239,68,68,.07);
    border:1px solid rgba(239,68,68,.25);
    border-radius:12px;
    padding:14px 18px;
    margin-top:20px;
    font-size:.78rem;
    color:#fca5a5;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md   = os.path.join(base, "models")
    model  = joblib.load(os.path.join(md, "best_model.joblib"))
    scaler = joblib.load(os.path.join(md, "scaler_v2.joblib"))
    feats  = joblib.load(os.path.join(md, "feature_names_v2.joblib"))
    with open(os.path.join(md, "best_model_name.json")) as f: meta = json.load(f)
    with open(os.path.join(md, "shap_info.json")) as f:      si   = json.load(f)
    return model, scaler, feats, meta, si

model, scaler, FEATURES, meta, shap_info = load_artifacts()
METRICS    = meta["metrics"]
THRESHOLD  = meta.get("threshold", 0.5)
MODEL_NAME = meta["name"]
GLOBAL_SHAP= dict(zip(shap_info["feature_names"], shap_info["mean_abs_shap"]))

def eng_features(vals: dict) -> list:
    """Apply same feature engineering as training."""
    g  = vals["Glucose"]; b = vals["BMI"]; a = vals["Age"]
    i  = vals["Insulin"]; bp= vals["BloodPressure"]
    sk = vals["SkinThickness"]; p = vals["DiabetesPedigreeFunction"]
    pr = vals["Pregnancies"]
    return [
        pr, g, bp, sk, i, b, p, a,
        g*b, g*a, b*a,
        i/(g+1e-6), i/(b+1e-6),
        int(g>=126), int(b>=30), int(bp>=80), int(a>=45), int(i>=166),
        g**2, b**2, a**2,
        p*a
    ]

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🩺 <span>DiabetesAI</span> — Risk Predictor</h1>
  <p>Stacking ensemble model · XGBoost + LightGBM + Random Forest + Extra Trees · SHAP explainability</p>
</div>
""", unsafe_allow_html=True)

# ── Model metrics banner ───────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
with c1: st.metric("🏆 Best Model",  MODEL_NAME.split()[0])
with c2: st.metric("🎯 Recall",      f"{METRICS['recall']:.1%}")
with c3: st.metric("📈 ROC-AUC",     f"{METRICS['roc_auc']:.1%}")
with c4: st.metric("✅ Accuracy",    f"{METRICS['accuracy']:.1%}")

st.markdown("---")

# ── Two-column layout ──────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ── LEFT: inputs ───────────────────────────────────────────────────────────
FEATURE_META = {
    "Pregnancies":             ("🤰 Pregnancies",               0,   17,  3,   1,   ""),
    "Glucose":                 ("🍭 Plasma Glucose",            44,  199, 120, 1,   "mg/dL"),
    "BloodPressure":           ("❤️ Diastolic BP",             24,  122, 72,  1,   "mmHg"),
    "SkinThickness":           ("📏 Skin Thickness",            7,   99,  23,  1,   "mm"),
    "Insulin":                 ("💉 Serum Insulin",             14,  846, 80,  5,   "µU/mL"),
    "BMI":                     ("⚖️ Body Mass Index (kg/m²)",  18,  67,  32,  0.1, ""),
    "DiabetesPedigreeFunction":("🧬 Diabetes Pedigree Function",0.08,2.42,0.47,0.01,""),
    "Age":                     ("🎂 Age",                       21,  81,  33,  1,   "years"),
}

with left:
    st.markdown('<div class="section-hdr">📋 Enter Health Metrics</div>', unsafe_allow_html=True)
    input_vals = {}
    ORIG_FEATS = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                  "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    for feat in ORIG_FEATS:
        label, mn, mx, dv, step, unit = FEATURE_META[feat]
        lbl = f"{label} ({unit})" if unit else label
        input_vals[feat] = st.slider(lbl, float(mn), float(mx), float(dv), float(step))

    predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True)

# ── RIGHT: results ─────────────────────────────────────────────────────────
with right:
    if not predict_btn:
        # Radar chart of default values (normalised 0-1)
        st.markdown('<div class="section-hdr">📊 Feature Overview</div>', unsafe_allow_html=True)
        norm_ranges = {"Pregnancies":(0,17),"Glucose":(44,199),"BloodPressure":(24,122),
                       "SkinThickness":(7,99),"Insulin":(14,846),"BMI":(18,67),
                       "DiabetesPedigreeFunction":(0.08,2.42),"Age":(21,81)}
        radar_vals  = [(input_vals.get(f,0)-lo)/(hi-lo)
                       for f,(lo,hi) in norm_ranges.items()]
        radar_labels= ["Pregnancies","Glucose","Blood\nPressure","Skin\nThickness",
                       "Insulin","BMI","Pedigree","Age"]
        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals+[radar_vals[0]], theta=radar_labels+[radar_labels[0]],
            fill="toself", fillcolor="rgba(99,102,241,0.15)",
            line=dict(color="#6366f1", width=2),
            marker=dict(color="#a78bfa", size=6)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(255,255,255,.08)",
                                tickfont=dict(color="#64748b", size=9)),
                angularaxis=dict(gridcolor="rgba(255,255,255,.08)",
                                 tickfont=dict(color="#94a3b8", size=10))
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60,r=60,t=20,b=20), height=340,
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Global SHAP importance bar
        st.markdown('<div class="section-hdr">🧠 Global Feature Importance (SHAP)</div>',
                    unsafe_allow_html=True)
        top_global = sorted(GLOBAL_SHAP.items(), key=lambda x:x[1], reverse=True)[:8]
        names_g = [x[0] for x in top_global]
        vals_g  = [x[1] for x in top_global]
        fig_bar = go.Figure(go.Bar(
            x=vals_g, y=names_g, orientation="h",
            marker=dict(
                color=vals_g,
                colorscale=[[0,"#6366f1"],[0.5,"#a78bfa"],[1,"#f472b6"]],
                showscale=False
            ),
            text=[f"{v:.3f}" for v in vals_g], textposition="outside",
            textfont=dict(color="#94a3b8", size=10)
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.06)",
                       tickfont=dict(color="#64748b"),title=dict(text="Mean |SHAP|", font=dict(color="#64748b"))),
            yaxis=dict(tickfont=dict(color="#94a3b8")),
            margin=dict(l=10,r=60,t=10,b=10), height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        # ── PREDICTION ─────────────────────────────────────────────────────
        eng = eng_features(input_vals)
        inp_scaled = scaler.transform([eng])
        prob  = model.predict_proba(inp_scaled)[0, 1]
        pred  = int(prob >= THRESHOLD)

        if prob < 0.35:
            tier, tc, rc = "LOW RISK",      "#10b981", "result-low"
        elif prob < 0.60:
            tier, tc, rc = "MODERATE RISK", "#f59e0b", "result-medium"
        else:
            tier, tc, rc = "HIGH RISK",     "#ef4444", "result-high"

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob*100, 1),
            delta={"reference": 35, "valueformat":".1f",
                   "increasing":{"color":"#ef4444"}, "decreasing":{"color":"#10b981"}},
            gauge={
                "axis":{"range":[0,100], "tickwidth":1,
                        "tickcolor":"rgba(255,255,255,.2)",
                        "tickfont":{"color":"#64748b"}},
                "bar":{"color": tc, "thickness":.25},
                "bgcolor":"rgba(0,0,0,0)",
                "borderwidth":0,
                "steps":[
                    {"range":[0,35],  "color":"rgba(16,185,129,.12)"},
                    {"range":[35,60], "color":"rgba(245,158,11,.12)"},
                    {"range":[60,100],"color":"rgba(239,68,68,.12)"},
                ],
                "threshold":{"line":{"color":"white","width":3},"thickness":.85,"value":prob*100},
            },
            number={"suffix":"%","font":{"size":48,"color":tc,"family":"Inter"}},
            title={"text":f"<b>{tier}</b>","font":{"size":14,"color":tc}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30,r=30,t=30,b=0), height=260
        )
        st.markdown('<div class="section-hdr">📊 Risk Assessment</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SHAP individual ────────────────────────────────────────────────
        try:
            inp_df = pd.DataFrame(inp_scaled, columns=FEATURES)
            xgb_comp = model.named_estimators_["xgb"]
            expl = shap.TreeExplainer(xgb_comp)
            sv   = expl(inp_df).values
            if sv.ndim == 3: sv = sv[:, :, 1]
            sv   = sv[0]

            # Top 5 drivers
            top5 = sorted(enumerate(sv), key=lambda x: abs(x[1]), reverse=True)[:5]

            st.markdown('<div class="section-hdr">🧠 What Drove This Prediction?</div>',
                        unsafe_allow_html=True)

            ICONS = {"Glucose":"🍭","BMI":"⚖️","Age":"🎂","Pregnancies":"🤰",
                     "Insulin":"💉","BloodPressure":"❤️","SkinThickness":"📏",
                     "DiabetesPedigreeFunction":"🧬","Glucose_BMI":"🍭⚖️",
                     "Glucose_Age":"🍭🎂","BMI_Age":"⚖️🎂","Insulin_Glucose_ratio":"💉🍭",
                     "HighGlucose":"⚠️","HighBMI":"⚠️","Glucose2":"🍭²","BMI2":"⚖️²"}

            max_abs = max(abs(sv[i]) for i,_ in top5) or 1
            shap_bar_vals = []
            shap_bar_names= []
            shap_bar_colors=[]

            for idx, sv_val in top5:
                feat   = FEATURES[idx]
                icon   = ICONS.get(feat, "📌")
                up     = sv_val > 0
                colour = "#ef4444" if up else "#10b981"
                arrow  = "↑ increases risk" if up else "↓ decreases risk"
                pct    = min(100, abs(sv_val)/max_abs*100)

                # Get human-readable value
                orig_feat = feat if feat in input_vals else None
                val_txt   = f"{input_vals[orig_feat]:.1f}" if orig_feat else "—"

                st.markdown(f"""
                <div class="driver">
                  <div class="driver-icon">{icon}</div>
                  <div style="flex:1.4">
                    <div class="driver-name">{feat.replace('_',' ')}</div>
                    <div class="driver-desc">Value: {val_txt} &nbsp;·&nbsp; {arrow}</div>
                  </div>
                  <div class="driver-bar-wrap" style="flex:.7">
                    <div class="driver-bar" style="width:{pct:.0f}%;background:{colour};"></div>
                  </div>
                  <div style="font-size:.8rem;color:{colour};font-weight:700;min-width:52px;text-align:right">
                    {sv_val:+.3f}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                shap_bar_names.append(feat.replace("_"," "))
                shap_bar_vals.append(sv_val)
                shap_bar_colors.append("#ef4444" if up else "#10b981")

            # Mini SHAP bar chart
            fig_shap = go.Figure(go.Bar(
                x=shap_bar_vals, y=shap_bar_names, orientation="h",
                marker_color=shap_bar_colors, opacity=0.85,
                text=[f"{v:+.3f}" for v in shap_bar_vals],
                textposition="outside", textfont=dict(color="#94a3b8", size=10)
            ))
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.05)",
                           zeroline=True, zerolinecolor="rgba(255,255,255,.2)",
                           tickfont=dict(color="#64748b"),
                           title=dict(text="SHAP value", font=dict(color="#64748b"))),
                yaxis=dict(tickfont=dict(color="#94a3b8")),
                margin=dict(l=10,r=60,t=10,b=10), height=220
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        except Exception as e:
            st.info(f"SHAP could not compute for this input: {e}")

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Medical Disclaimer:</b> This tool is for educational & research purposes only —
        not a substitute for professional medical advice. Always consult a qualified
        healthcare provider.
        </div>""", unsafe_allow_html=True)

# ── About expander ─────────────────────────────────────────────────────────
with st.expander("📖 About this model & methodology"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Dataset**: Pima Indians Diabetes Database (768 samples, 8 features)
**Architecture**: Stacking Ensemble
- Base: XGBoost (Optuna-tuned) + LightGBM (Optuna-tuned) + Random Forest + Extra Trees + Gradient Boosting
- Meta-learner: Logistic Regression
- Feature engineering: 22 features (14 engineered from 8 original)
- SMOTE oversampling to fix class imbalance
- Decision threshold optimised for Recall
""")
    with c2:
        st.markdown(f"""
| Metric | Score |
|--------|-------|
| Accuracy  | {METRICS['accuracy']:.3f} |
| Precision | {METRICS['precision']:.3f} |
| **Recall** | **{METRICS['recall']:.3f}** |
| F1 Score  | {METRICS['f1']:.3f} |
| ROC-AUC   | {METRICS['roc_auc']:.3f} |
| Threshold | {THRESHOLD:.2f} |

**Priority metric: Recall** — false negatives (missed diagnoses) are far more
costly than false positives in a medical screening context.
""")
