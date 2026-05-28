import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Load and clean
df = pd.read_csv('Life Expectancy Data.csv')
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
df.dropna(subset=['Life expectancy'], inplace=True)

# Impute
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Encode
le = LabelEncoder()
df['Status'] = le.fit_transform(df['Status'].astype(str))

# Transform skewed features
positive_skew = ['Population', 'infant deaths', 'under-five deaths',
                 'Measles', 'HIV/AIDS', 'percentage expenditure',
                 'GDP', 'thinness 5-9 years', 'thinness 1-19 years',
                 'Adult Mortality']
for col in positive_skew:
    df[col] = np.log1p(df[col])
df['HIV/AIDS'] = np.sqrt(df['HIV/AIDS'])

# Features and target
X = df.drop(columns=['Country', 'Life expectancy'])
y = df['Life expectancy']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train and save models
models = {
    'linear_regression' : LinearRegression(),
    'decision_tree'     : DecisionTreeRegressor(max_depth=9, random_state=42),
    'random_forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
    'svr'               : SVR(kernel='rbf', C=10, epsilon=0.1),
    'xgboost'           : XGBRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, random_state=42, verbosity=0)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f'{name}.pkl')
    print(f'Saved: {name}.pkl')

# Save scaler and feature columns
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_cols.pkl')
print('Saved: scaler.pkl')
print('Saved: feature_cols.pkl')
print('All done!')