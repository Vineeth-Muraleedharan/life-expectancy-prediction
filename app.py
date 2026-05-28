import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Life Expectancy Prediction", page_icon="🏥", layout="wide")

# Load models and scaler
scaler       = joblib.load('scaler.pkl')
feature_cols = joblib.load('feature_cols.pkl')

models = {
    'Linear Regression' : joblib.load('linear_regression.pkl'),
    'Decision Tree'     : joblib.load('decision_tree.pkl'),
    'Random Forest'     : joblib.load('random_forest.pkl'),
    'SVR'               : joblib.load('svr.pkl'),
    'XGBoost'           : joblib.load('xgboost.pkl')
}

# Header
st.title("🏥 Life Expectancy Prediction")
st.markdown("**WHO Dataset · Regression · 2,928 records · 20 features**")
st.markdown("---")

st.markdown("### Enter Details to Predict Life Expectancy")
st.markdown("Adjust the values below and click **Predict**.")
st.markdown("---")

# Input sliders
c1, c2, c3 = st.columns(3)
with c1:
    status      = st.selectbox("Country Status", [0, 1],
                    format_func=lambda x: "Developing" if x==0 else "Developed")
    schooling   = st.slider("Schooling (years)", 0.0, 20.0, 12.0)
    income_comp = st.slider("Income Composition", 0.0, 1.0, 0.6)
    bmi         = st.slider("BMI", 1.0, 80.0, 25.0)
    adult_mort  = st.slider("Adult Mortality", 1, 800, 150)
with c2:
    hiv        = st.slider("HIV/AIDS", 0.0, 50.0, 0.5)
    gdp        = st.slider("GDP (USD)", 0, 100000, 5000)
    alcohol    = st.slider("Alcohol", 0.0, 20.0, 5.0)
    polio      = st.slider("Polio %", 0, 100, 80)
    diphtheria = st.slider("Diphtheria %", 0, 100, 80)
with c3:
    hepatitis_b   = st.slider("Hepatitis B %", 0, 100, 75)
    infant_deaths = st.slider("Infant deaths", 0, 200, 30)
    population    = st.slider("Population (M)", 0.1, 1400.0, 10.0)
    thinness      = st.slider("Thinness 1-19 yrs %", 0.0, 30.0, 5.0)
    year          = st.slider("Year", 2000, 2015, 2010)

# Predict button
if st.button("🔮 Predict Life Expectancy", use_container_width=True):
    inp = {
        'Year'                           : year,
        'Status'                         : status,
        'Adult Mortality'                : np.log1p(adult_mort),
        'infant deaths'                  : np.log1p(infant_deaths),
        'Alcohol'                        : alcohol,
        'percentage expenditure'         : 0.0,
        'Hepatitis B'                    : hepatitis_b,
        'Measles'                        : np.log1p(0),
        'BMI'                            : bmi,
        'under-five deaths'              : np.log1p(0),
        'Polio'                          : polio,
        'Total expenditure'              : 6.0,
        'Diphtheria'                     : diphtheria,
        'HIV/AIDS'                       : np.sqrt(np.log1p(hiv)),
        'GDP'                            : np.log1p(gdp),
        'Population'                     : np.log1p(population * 1e6),
        'thinness 1-19 years'            : thinness,
        'thinness 5-9 years'             : thinness,
        'Income composition of resources': income_comp,
        'Schooling'                      : schooling
    }

    inp_df     = pd.DataFrame([inp])[feature_cols]
    inp_scaled = scaler.transform(inp_df)

    st.markdown("---")
    st.markdown("### 🏆 Predicted Life Expectancy — All Models")

    cols = st.columns(5)
    for i, (name, model) in enumerate(models.items()):
        pred = model.predict(inp_scaled)[0]
        cols[i].metric(name, f"{pred:.1f} yrs")