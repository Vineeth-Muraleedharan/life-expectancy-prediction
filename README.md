# 🏥 Life Expectancy Prediction App

A machine learning web app that predicts life expectancy based on 
health, economic, and demographic indicators from the WHO dataset.

## 🔗 Live App
[Life Expectancy Prediction](https://life-expectancy-prediction.streamlit.app)

## 📊 Dataset
- Source: WHO Life Expectancy Dataset (Kaggle)
- Records: 2,928 | Features: 20
- Target: Life Expectancy (years)

## 🤖 Models Used
| Model | Test R² | MAE |
|---|---|---|
| Random Forest | 0.9674 | 1.05 yrs |
| XGBoost | 0.9564 | 1.34 yrs |
| SVR | 0.9379 | 1.54 yrs |
| Decision Tree | 0.9354 | 1.55 yrs |
| Linear Regression | 0.8424 | 2.84 yrs |

## 📁 Project Structure
| File | Description |
|---|---|
| `app.py` | Streamlit prediction app |
| `train_model.py` | Model training script |
| `code.ipynb` | Full ML pipeline notebook |
| `scaler.pkl` | Saved StandardScaler |
| `*.pkl` | Saved trained models |
| `requirements.txt` | Dependencies |

## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 🛠️ Tech Stack
Python · Streamlit · Scikit-learn · XGBoost · Pandas · NumPy · Joblib

## 📌 Key Findings
- **Best Model:** Random Forest (R² = 0.9674)
- **Top Predictors:** HIV/AIDS, Income Composition, Adult Mortality
- **Prediction accuracy:** within ~1 year of actual life expectancy
