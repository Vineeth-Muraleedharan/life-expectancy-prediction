import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Life Expectancy Prediction", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.stApp {
    background-image: url("https://www.techtarget.com/rms/onlineimages/machine%20leaning_g1150854211.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.block-container {
    background: rgba(0, 0, 0, 0.75);
    padding: 2rem 3rem;
    border-radius: 16px;
}
h1 {
    color: #00C9A7 !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    text-align: center;
}
h3 { color: #ffffff !important; text-align: center; }
.subtitle {
    text-align: center;
    color: #cccccc;
    font-size: 16px;
    margin-bottom: 1rem;
}
.stSlider label, .stSelectbox label,
.stNumberInput label, .stRadio label {
    color: #ffffff !important;
    font-weight: 500 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00C9A7, #0077B6);
    color: white;
    font-size: 18px;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    width: 100%;
    transition: 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0077B6, #00C9A7);
    transform: scale(1.02);
}
.section-header {
    color: #00C9A7;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 1.5rem 0 0.5rem;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(0,201,167,0.3);
}
.result-card {
    background: rgba(0, 201, 167, 0.15);
    border: 1px solid #00C9A7;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin: 0.5rem 0;
}
.result-model { color: #aaaaaa; font-size: 13px; margin-bottom: 4px; }
.result-value { color: #00C9A7; font-size: 32px; font-weight: 800; }
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
scaler       = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_cols.pkl')
models = {
    'Linear Regression': joblib.load('linear_regression.pkl'),
    'Decision Tree'    : joblib.load('decision_tree.pkl'),
    'Random Forest'    : joblib.load('random_forest.pkl'),
    'SVR'              : joblib.load('svr.pkl'),
    'XGBoost'          : joblib.load('xgboost.pkl')
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🏥 Life Expectancy Prediction")
st.markdown('<p class="subtitle">WHO Dataset · Regression · 2,928 records · 20 features · 5 ML Models</p>',
            unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("### ⚙️ Enter Country Details")
st.markdown("")

# ── Section 1 — Country Info ──────────────────────────────────────────────────
st.markdown('<p class="section-header">🌍 Country Information</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    status = st.selectbox("Country Status", ["Developing", "Developed"])
with c2:
    year = st.selectbox("Year", list(range(2015, 1999, -1)))
with c3:
    schooling = st.selectbox("Avg Schooling (years)",
                             [f"{x:.1f}" for x in np.arange(0, 20.5, 0.5)])

# ── Section 2 — Economic Indicators ──────────────────────────────────────────
st.markdown('<p class="section-header">💰 Economic Indicators</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    gdp = st.number_input("GDP per capita (USD)", min_value=0,
                          max_value=100000, value=5000, step=100)
with c2:
    income_comp = st.selectbox("Income Composition of Resources",
                               [f"{x:.2f}" for x in np.arange(0.0, 1.05, 0.05)])
with c3:
    total_exp = st.number_input("Total Health Expenditure (%)", min_value=0.0,
                                max_value=20.0, value=6.0, step=0.1)

# ── Section 3 — Health Indicators ────────────────────────────────────────────
st.markdown('<p class="section-header">🏥 Health Indicators</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    bmi          = st.number_input("Average BMI", min_value=1.0,
                                   max_value=80.0, value=25.0, step=0.1)
    adult_mort   = st.number_input("Adult Mortality (per 1000)",
                                   min_value=1, max_value=800, value=150)
with c2:
    hiv          = st.number_input("HIV/AIDS Deaths (per 1000)",
                                   min_value=0.0, max_value=50.0, value=0.5, step=0.1)
    alcohol      = st.number_input("Alcohol Consumption (litres/capita)",
                                   min_value=0.0, max_value=20.0, value=5.0, step=0.1)
with c3:
    infant_deaths = st.number_input("Infant Deaths (per 1000)",
                                    min_value=0, max_value=200, value=30)
    thinness      = st.number_input("Thinness 1-19 years (%)",
                                    min_value=0.0, max_value=30.0, value=5.0, step=0.1)

# ── Section 4 — Immunisation Coverage ────────────────────────────────────────
st.markdown('<p class="section-header">💉 Immunisation Coverage (%)</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    polio = st.select_slider("Polio", options=list(range(0, 101, 1)), value=80)
with c2:
    diphtheria = st.select_slider("Diphtheria", options=list(range(0, 101, 1)), value=80)
with c3:
    hepatitis_b = st.select_slider("Hepatitis B", options=list(range(0, 101, 1)), value=75)

# ── Section 5 — Demographics ──────────────────────────────────────────────────
st.markdown('<p class="section-header">👥 Demographics</p>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    population = st.number_input("Population (Millions)",
                                 min_value=0.1, max_value=1400.0, value=10.0, step=0.1)
with c2:
    pct_exp = st.number_input("Health % of GDP Expenditure",
                              min_value=0.0, max_value=20000.0, value=100.0, step=10.0)

# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🔮 Predict Life Expectancy")

if predict_btn:
    inp = {
        'Year'                           : int(year),
        'Status'                         : 1 if status == "Developed" else 0,
        'Adult Mortality'                : np.log1p(adult_mort),
        'infant deaths'                  : np.log1p(infant_deaths),
        'Alcohol'                        : alcohol,
        'percentage expenditure'         : np.log1p(pct_exp),
        'Hepatitis B'                    : hepatitis_b,
        'Measles'                        : np.log1p(0),
        'BMI'                            : bmi,
        'under-five deaths'              : np.log1p(0),
        'Polio'                          : polio,
        'Total expenditure'              : total_exp,
        'Diphtheria'                     : diphtheria,
        'HIV/AIDS'                       : np.sqrt(np.log1p(hiv)),
        'GDP'                            : np.log1p(gdp),
        'Population'                     : np.log1p(population * 1e6),
        'thinness 1-19 years'            : thinness,
        'thinness 5-9 years'             : thinness,
        'Income composition of resources': float(income_comp),
        'Schooling'                      : float(schooling)
    }

    inp_df     = pd.DataFrame([inp])[feature_cols]
    inp_scaled = scaler.transform(inp_df)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### 🏆 Predicted Life Expectancy — All Models")
    st.markdown("")

    cols = st.columns(5)
    for i, (name, model) in enumerate(models.items()):
        pred = model.predict(inp_scaled)[0]
        cols[i].markdown(f"""
        <div class="result-card">
            <div class="result-model">{name}</div>
            <div class="result-value">{pred:.1f} yrs</div>
        </div>
        """, unsafe_allow_html=True)

    preds    = {name: model.predict(inp_scaled)[0] for name, model in models.items()}
    best_pred = preds['Random Forest']
    st.markdown("")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(0,201,167,0.2), rgba(0,119,182,0.2));
                border: 1px solid #00C9A7; border-radius: 12px;
                padding: 1.5rem; text-align: center; margin-top: 1rem;">
        <div style="color: #aaaaaa; font-size: 14px;">🏆 Best Model (Random Forest)</div>
        <div style="color: #00C9A7; font-size: 48px; font-weight: 800;">{best_pred:.1f} years</div>
        <div style="color: #cccccc; font-size: 13px;">R² = 0.9674 · MAE = 1.05 years</div>
    </div>
    """, unsafe_allow_html=True)