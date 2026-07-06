import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Term Deposit Predictor",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────
#  LOAD ARTIFACTS  (keys verified from pkl)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    arts = joblib.load("model_prod_files1.pkl")
    return (
        arts["model"],
        arts["cat_encoder"],
        arts["num_scaler"],
        arts["label_encoder"]
    )

rf_model, OHE, scaler, LE = load_artifacts()

# ─────────────────────────────────────────────────────────────
#  COLUMN DEFINITIONS  (verified from pkl)
# ─────────────────────────────────────────────────────────────
CAT_COLS = [
    "job","marital","education","housing","loan",
    "contact","month","day_of_week","poutcome"
]
NUM_COLS = [
    "age","duration","campaign","emp_var_rate",
    "cons_price_idx","cons_conf_idx","euribor3m","nr_employed"
]
NO_TRANS_COLS = ["previous","previously_contacted"]

# ─────────────────────────────────────────────────────────────
#  EXACT CATEGORIES  (verified from OHE.categories_)
# ─────────────────────────────────────────────────────────────
JOB_OPTS  = ["admin","blue-collar","entrepreneur","housemaid","management",
             "retired","self-employed","services","student","technician","unemployed"]
EDU_OPTS  = ["basic_4y","basic_6y","basic_9y","high_school",
             "illiterate","professional_course","university_degree"]
MON_OPTS  = ["apr","aug","dec","jul","jun","mar","may","nov","oct","sep"]
DAY_OPTS  = ["fri","mon","thu","tue","wed"]
POUT_OPTS = ["failure","nonexistent","success"]

# ─────────────────────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }
.stApp { background: #06091a; }
.block-container { padding: 0 2.5rem 3rem !important; max-width: 1200px !important; }

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1f 0%, #0a1028 100%);
    border-right: 1px solid rgba(99,120,220,0.12);
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,120,220,0.12);
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 7px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.stat-card .sk { font-size: 11px; color: #5a6480; }
.stat-card .sv { font-size: 11px; color: #7eb3ff;
                 font-family: 'JetBrains Mono', monospace; font-weight: 500; }

/* ══ HERO ══ */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #0f2050 40%, #0a1835 100%);
    border: 1px solid rgba(79,142,247,0.15);
    border-radius: 20px;
    padding: 36px 40px;
    margin: 24px 0 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';position:absolute;top:-60px;right:-60px;
    width:260px;height:260px;border-radius:50%;
    background:radial-gradient(circle,rgba(79,142,247,0.08) 0%,transparent 70%);
}
.hero::after {
    content:'';position:absolute;bottom:-40px;left:200px;
    width:200px;height:200px;border-radius:50%;
    background:radial-gradient(circle,rgba(61,214,140,0.05) 0%,transparent 70%);
}
.hero-badge {
    display:inline-flex;align-items:center;gap:6px;
    background:rgba(79,142,247,0.1);border:1px solid rgba(79,142,247,0.25);
    border-radius:20px;padding:5px 14px;
    font-size:10px;font-weight:700;color:#7eb3ff;
    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;
}
.hero h1 {
    font-size:26px;font-weight:800;color:#e8ecf8;
    margin:0 0 10px;letter-spacing:-0.03em;line-height:1.2;
}
.hero h1 span { color:#4f8ef7; }
.hero p { font-size:13px;color:#6272a4;margin:0;line-height:1.6;max-width:520px; }
.hero-pills { display:flex;gap:8px;margin-top:18px;flex-wrap:wrap; }
.pill { display:inline-flex;align-items:center;gap:5px;
        padding:4px 12px;border-radius:20px;font-size:11px;font-weight:500; }
.pill-blue   { background:rgba(79,142,247,0.1); color:#7eb3ff; border:1px solid rgba(79,142,247,0.2); }
.pill-green  { background:rgba(61,214,140,0.1); color:#3dd68c; border:1px solid rgba(61,214,140,0.2); }
.pill-purple { background:rgba(155,109,255,0.1);color:#b48eff; border:1px solid rgba(155,109,255,0.2); }
.pill-amber  { background:rgba(255,201,71,0.1); color:#ffc947; border:1px solid rgba(255,201,71,0.2); }

/* ══ FORM ══ */
.form-card-title {
    font-size:11px;font-weight:700;letter-spacing:0.12em;
    text-transform:uppercase;color:#4f8ef7;
    display:flex;align-items:center;gap:8px;
    margin-bottom:20px;padding-bottom:12px;
    border-bottom:1px solid rgba(79,142,247,0.1);
}
.form-card-title::before {
    content:'';width:3px;height:14px;
    background:linear-gradient(to bottom,#4f8ef7,#3dd68c);
    border-radius:2px;display:inline-block;
}
.custom-divider {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(79,142,247,0.2),transparent);
    margin:8px 0 24px;border:none;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"]   label {
    font-size:11px !important;font-weight:500 !important;
    color:#5a6480 !important;letter-spacing:0.03em !important;
    text-transform:uppercase !important;margin-bottom:4px !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background:#080d1f !important;
    border:1px solid rgba(99,120,220,0.2) !important;
    border-radius:8px !important;
    color:#c8d0e8 !important;font-size:13px !important;font-weight:500 !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color:rgba(79,142,247,0.5) !important;
    box-shadow:0 0 0 3px rgba(79,142,247,0.07) !important;
}

/* ══ BUTTON ══ */
div[data-testid="stForm"] .stButton > button {
    background:linear-gradient(135deg,#1a4fc4 0%,#4f8ef7 100%) !important;
    color:#fff !important;font-size:14px !important;font-weight:700 !important;
    border:none !important;border-radius:12px !important;
    padding:0.8rem 2rem !important;width:100% !important;
    letter-spacing:0.04em !important;
    box-shadow:0 6px 28px rgba(79,142,247,0.3) !important;
    transition:all 0.2s !important;
}
div[data-testid="stForm"] .stButton > button:hover {
    box-shadow:0 8px 36px rgba(79,142,247,0.45) !important;
    transform:translateY(-1px) !important;
}

/* ══ RESULT CARDS ══ */
.result-yes {
    background:linear-gradient(135deg,#041a0f,#071f12);
    border:1px solid rgba(61,214,140,0.3);border-radius:16px;
    padding:28px 32px;position:relative;overflow:hidden;
}
.result-yes::before {
    content:'';position:absolute;top:-30px;right:-30px;
    width:120px;height:120px;border-radius:50%;
    background:radial-gradient(circle,rgba(61,214,140,0.08) 0%,transparent 70%);
}
.result-no {
    background:linear-gradient(135deg,#1a0404,#1f0707);
    border:1px solid rgba(255,107,107,0.3);border-radius:16px;
    padding:28px 32px;position:relative;overflow:hidden;
}
.result-no::before {
    content:'';position:absolute;top:-30px;right:-30px;
    width:120px;height:120px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,107,107,0.08) 0%,transparent 70%);
}
.result-icon { font-size:40px;margin-bottom:12px; }
.result-title { font-size:22px;font-weight:800;letter-spacing:-0.02em;margin:0 0 8px; }
.result-yes .result-title { color:#3dd68c; }
.result-no  .result-title { color:#ff6b6b; }
.result-desc { font-size:13px;color:#6272a4;line-height:1.6;margin:0; }
.conf-badge {
    display:inline-flex;align-items:center;gap:6px;
    padding:6px 14px;border-radius:20px;
    font-size:11px;font-weight:700;margin-top:14px;
}
.conf-high   { background:rgba(61,214,140,0.1); color:#3dd68c; border:1px solid rgba(61,214,140,0.25); }
.conf-medium { background:rgba(255,201,71,0.1); color:#ffc947; border:1px solid rgba(255,201,71,0.25); }
.conf-low    { background:rgba(255,107,107,0.1);color:#ff6b6b; border:1px solid rgba(255,107,107,0.25); }

/* ══ PROBABILITY CARD ══ */
.prob-card {
    background:linear-gradient(135deg,#0b1022,#0d1428);
    border:1px solid rgba(99,120,220,0.12);border-radius:16px;
    padding:24px;height:100%;
}
.prob-title { font-size:11px;font-weight:700;letter-spacing:0.1em;
              text-transform:uppercase;color:#5a6480;margin-bottom:18px; }
.prob-bar-wrap { background:rgba(255,255,255,0.04);border-radius:6px;
                 height:8px;overflow:hidden;margin:8px 0 4px; }
.prob-bar-yes { height:100%;border-radius:6px;
                background:linear-gradient(90deg,#1a7c4f,#3dd68c); }
.prob-bar-no  { height:100%;border-radius:6px;
                background:linear-gradient(90deg,#7c1a1a,#ff6b6b); }
.prob-row { display:flex;justify-content:space-between;align-items:center;
            font-size:12px;margin-bottom:14px; }
.prob-label { color:#6272a4; }
.prob-value { font-weight:700;font-family:'JetBrains Mono',monospace;font-size:14px; }
.prob-yes { color:#3dd68c; }
.prob-no  { color:#ff6b6b; }

/* ══ SUMMARY ══ */
.summary-row {
    display:flex;justify-content:space-between;align-items:center;
    padding:7px 0;border-bottom:1px solid rgba(99,120,220,0.07);font-size:12px;
}
.summary-row:last-child { border-bottom:none; }
.summary-key { color:#5a6480; }
.summary-val { color:#c8d0e8;font-family:'JetBrains Mono',monospace;font-size:11px; }
.info-note {
    background:rgba(79,142,247,0.04);
    border:1px solid rgba(79,142,247,0.12);
    border-left:3px solid rgba(79,142,247,0.5);
    border-radius:8px;padding:12px 16px;
    font-size:12px;color:#6272a4;line-height:1.7;margin-top:20px;
}
.info-note strong { color:#7eb3ff; }
[data-testid="stExpander"] {
    background:#0b1022 !important;
    border:1px solid rgba(99,120,220,0.12) !important;
    border-radius:12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 8px 20px;text-align:center;'>
        <div style='width:56px;height:56px;background:linear-gradient(135deg,#1a52c4,#4f8ef7);
                    border-radius:14px;display:flex;align-items:center;justify-content:center;
                    font-size:26px;margin:0 auto 12px;
                    box-shadow:0 6px 20px rgba(79,142,247,0.3);'>🏦</div>
        <div style='font-size:15px;font-weight:700;color:#e8ecf8;'>Term Deposit</div>
        <div style='font-size:15px;font-weight:700;color:#4f8ef7;'>Predictor</div>
        <div style='font-size:11px;color:#3a4060;margin-top:6px;'>Bank Marketing · ML Dashboard</div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(99,120,220,0.1);margin:0 0 16px;'/>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a4060;margin-bottom:10px;padding:0 4px;'>Model Details</div>", unsafe_allow_html=True)
    for k, v in [
        ("Algorithm",   "Random Forest"),
        ("Estimators",  "100 trees"),
        ("Random State","42"),
        ("Dataset",     "Bank Marketing"),
        ("Total Rows",  "41,188"),
        ("Features",    "20 columns"),
        ("Train Split", "80% → 32,950"),
        ("Test Split",  "20% → 8,238"),
        ("Target",      "y → yes / no"),
        ("LE Classes",  "no=0 · yes=1"),
    ]:
        st.markdown(f"<div class='stat-card'><span class='sk'>{k}</span><span class='sv'>{v}</span></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(99,120,220,0.1);margin:16px 0;'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3a4060;margin-bottom:10px;padding:0 4px;'>Preprocessing Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px;line-height:2.2;padding:0 4px;'>
        <div><span style='display:inline-block;width:10px;height:10px;background:#4f8ef7;border-radius:2px;margin-right:8px;vertical-align:middle;'></span><b style='color:#7eb3ff;'>One Hot Encoder</b> <span style='color:#5a6480;'>· 9 cols</span></div>
        <div><span style='display:inline-block;width:10px;height:10px;background:#ffc947;border-radius:2px;margin-right:8px;vertical-align:middle;'></span><b style='color:#ffc947;'>MinMax Scaler</b> <span style='color:#5a6480;'>· 8 cols</span></div>
        <div><span style='display:inline-block;width:10px;height:10px;background:#3dd68c;border-radius:2px;margin-right:8px;vertical-align:middle;'></span><b style='color:#3dd68c;'>No Transform</b> <span style='color:#5a6480;'>· 2 cols</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid rgba(99,120,220,0.1);margin:16px 0 10px;'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#2a3050;text-align:center;'>UCI ML Repository · Bank Marketing Dataset<br>© 2026 ML Project</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🏦 &nbsp; Bank Marketing Analytics</div>
    <h1>Will this customer subscribe to a<br><span>Term Deposit?</span></h1>
    <p>Enter customer demographic, contact, and economic details below.
       The Random Forest model will predict the likelihood of subscription.</p>
    <div class='hero-pills'>
        <span class='pill pill-blue'>⚡ Random Forest</span>
        <span class='pill pill-green'>✓ Binary Classification</span>
        <span class='pill pill-purple'>◈ 41,188 Records</span>
        <span class='pill pill-amber'>◎ 20 Features</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  FORM
# ─────────────────────────────────────────────────────────────
with st.form("predict_form", clear_on_submit=False):

    # Personal
    st.markdown("<div class='form-card-title'>👤 &nbsp; Personal Details</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age     = st.number_input("Age", min_value=18, max_value=95, value=35)
        marital = st.selectbox("Marital Status", ["married","single","divorced"])
    with c2:
        job       = st.selectbox("Occupation", JOB_OPTS)
        education = st.selectbox("Education Level", EDU_OPTS)
    with c3:
        housing = st.selectbox("Housing Loan", ["yes","no"])
        loan    = st.selectbox("Personal Loan", ["yes","no"])

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Contact
    st.markdown("<div class='form-card-title'>📞 &nbsp; Contact Details</div>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        contact     = st.selectbox("Contact Type", ["cellular","telephone"])
        month       = st.selectbox("Month of Last Contact", MON_OPTS)
    with c5:
        day_of_week = st.selectbox("Day of Week", DAY_OPTS)
        duration    = st.number_input("Call Duration (secs)", min_value=0, value=200)
    with c6:
        campaign = st.number_input("Contacts This Campaign", min_value=1, value=1)
        previous = st.number_input("Previous Contacts", min_value=0, value=0)
        poutcome = st.selectbox("Previous Outcome", POUT_OPTS)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Economic
    st.markdown("<div class='form-card-title'>📊 &nbsp; Economic Indicators</div>", unsafe_allow_html=True)
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        emp_var_rate   = st.number_input("Emp. Variation Rate",  value=1.1,    format="%.2f")
    with e2:
        cons_price_idx = st.number_input("Consumer Price Idx",   value=93.994, format="%.3f")
    with e3:
        cons_conf_idx  = st.number_input("Consumer Conf. Idx",   value=-36.4,  format="%.1f")
    with e4:
        euribor3m      = st.number_input("Euribor 3M Rate",      value=4.857,  format="%.3f")
    with e5:
        nr_employed    = st.number_input("Nr. Employed (000s)",  value=5191.0, format="%.1f")

    st.markdown("<br/>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍  Predict Subscription Likelihood", use_container_width=True)

# ─────────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────────
if submitted:

    with st.spinner("Analyzing customer profile..."):
        time.sleep(1.5)

    # previously_contacted — matches notebook: pdays==999 → 0 else 1
    previously_contacted = 1 if previous > 0 else 0

    # 1. OHE — cat columns in exact trained order
    cat_df = pd.DataFrame(
        [[job, marital, education, housing, loan,
          contact, month, day_of_week, poutcome]],
        columns=CAT_COLS
    )
    cat_trans = OHE.transform(cat_df)
    cat_trans.reset_index(drop=True, inplace=True)

    # 2. MinMaxScaler — num columns in exact trained order
    num_df = pd.DataFrame(
        [[age, duration, campaign, emp_var_rate,
          cons_price_idx, cons_conf_idx, euribor3m, nr_employed]],
        columns=NUM_COLS
    )
    num_trans = scaler.transform(num_df)
    num_trans.reset_index(drop=True, inplace=True)

    # 3. No-transform
    no_trans_df = pd.DataFrame(
        [[previous, previously_contacted]],
        columns=NO_TRANS_COLS
    )

    # 4. Concat: cat | num | no_trans  (exact notebook order)
    final_df = pd.concat([cat_trans, num_trans, no_trans_df], axis=1)

    # 5. Align to model feature order
    for col in rf_model.feature_names_in_:
        if col not in final_df.columns:
            final_df[col] = 0
    final_df = final_df[list(rf_model.feature_names_in_)]

    # 6. Predict
    pred       = rf_model.predict(final_df)[0]
    proba      = rf_model.predict_proba(final_df)[0]
    subscribed = int(pred) == 1          # LE: no=0, yes=1
    prob_yes   = round(float(proba[1]) * 100, 2)
    prob_no    = round(float(proba[0]) * 100, 2)

    dominant = max(prob_yes, prob_no)
    if dominant >= 80:
        conf_cls, conf_txt = "conf-high",   "High Confidence"
    elif dominant >= 60:
        conf_cls, conf_txt = "conf-medium", "Medium Confidence"
    else:
        conf_cls, conf_txt = "conf-low",    "Low Confidence"

    # ── Result UI ─────────────────────────────────────────────
    st.markdown("<hr style='border:none;border-top:1px solid rgba(99,120,220,0.1);margin:8px 0 24px;'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#4f8ef7;margin-bottom:18px;'>📋 &nbsp; Prediction Result</div>", unsafe_allow_html=True)

    res_col, prob_col = st.columns([1.1, 1], gap="large")

    with res_col:
        if subscribed:
            st.markdown(f"""
            <div class='result-yes'>
                <div class='result-icon'>✅</div>
                <div class='result-title'>Likely to Subscribe</div>
                <p class='result-desc'>This customer shows a strong likelihood of opening a
                term deposit. Prioritize this lead in your marketing campaign.</p>
                <div class='conf-badge {conf_cls}'>◉ &nbsp; {conf_txt} &nbsp;·&nbsp; {prob_yes}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-no'>
                <div class='result-icon'>❌</div>
                <div class='result-title'>Unlikely to Subscribe</div>
                <p class='result-desc'>This customer is unlikely to open a term deposit at
                this time. Consider focusing on higher-probability customers.</p>
                <div class='conf-badge {conf_cls}'>◉ &nbsp; {conf_txt} &nbsp;·&nbsp; {prob_no}%</div>
            </div>""", unsafe_allow_html=True)

    with prob_col:
        st.markdown(f"""
        <div class='prob-card'>
            <div class='prob-title'>Prediction Probability Breakdown</div>
            <div class='prob-row'>
                <span class='prob-label'>✅ Will Subscribe</span>
                <span class='prob-value prob-yes'>{prob_yes}%</span>
            </div>
            <div class='prob-bar-wrap'>
                <div class='prob-bar-yes' style='width:{prob_yes}%;'></div>
            </div>
            <div style='margin-top:14px;'></div>
            <div class='prob-row'>
                <span class='prob-label'>❌ Won't Subscribe</span>
                <span class='prob-value prob-no'>{prob_no}%</span>
            </div>
            <div class='prob-bar-wrap'>
                <div class='prob-bar-no' style='width:{prob_no}%;'></div>
            </div>
            <div style='margin-top:20px;padding-top:16px;border-top:1px solid rgba(99,120,220,0.1);'>
                <div style='font-size:10px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Model</div>
                <div style='font-size:12px;color:#7eb3ff;font-family:"JetBrains Mono",monospace;'>RandomForestClassifier</div>
                <div style='font-size:11px;color:#3a4060;margin-top:2px;'>n_estimators=100 · random_state=42</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Input Summary ─────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    with st.expander("📄  View Full Input Summary", expanded=False):
        summary = {
            "Age": age, "Job": job, "Marital Status": marital,
            "Education": education, "Housing Loan": housing,
            "Personal Loan": loan, "Contact Type": contact,
            "Month": month, "Day of Week": day_of_week,
            "Call Duration (s)": duration, "Campaign Contacts": campaign,
            "Previous Contacts": previous, "Previous Outcome": poutcome,
            "Emp. Variation Rate": emp_var_rate,
            "Consumer Price Idx": cons_price_idx,
            "Consumer Conf. Idx": cons_conf_idx,
            "Euribor 3M Rate": euribor3m,
            "Nr. Employed": nr_employed,
            "Previously Contacted": previously_contacted,
        }
        items = list(summary.items())
        half  = len(items) // 2
        sc1, sc2 = st.columns(2)
        with sc1:
            for k, v in items[:half]:
                st.markdown(f"<div class='summary-row'><span class='summary-key'>{k}</span><span class='summary-val'>{v}</span></div>", unsafe_allow_html=True)
        with sc2:
            for k, v in items[half:]:
                st.markdown(f"<div class='summary-row'><span class='summary-key'>{k}</span><span class='summary-val'>{v}</span></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-note'>
        ℹ️ &nbsp; Prediction powered by a <strong>Random Forest Classifier</strong> trained on the
        <strong>Bank Marketing Dataset</strong> (UCI ML Repository · 41,188 records).
        Economic indicators such as <strong>Euribor rate</strong> and
        <strong>employment variation rate</strong> carry high feature importance.
    </div>""", unsafe_allow_html=True)
