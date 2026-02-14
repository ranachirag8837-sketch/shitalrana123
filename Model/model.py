# ==========================================
# STUDENT RESULT PREDICTION - MODEL TRAINING
# ==========================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Dataset" / "student_data.csv"
MODEL_DIR = BASE_DIR / "Model"

df = pd.read_csv(DATA_PATH)

# Features & Target
X = df[["Study_Hours", "Attendance_Percentage",
        "Internal_Marks", "Assignment_Score"]]
y = df["Final_Marks"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "KNN": KNeighborsRegressor()
}

trained_models = {}
scores = {}

# Train
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    score = r2_score(y_test, preds)

    trained_models[name] = model
    scores[name] = score

    print(f"{name} R2 Score: {score:.4f}")

# Top 3 Models
top_models = sorted(scores, key=scores.get, reverse=True)[:3]

# Save Everything
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(trained_models, MODEL_DIR / "models.pkl")
joblib.dump(top_models, MODEL_DIR / "top_models.pkl")
joblib.dump(scores, MODEL_DIR / "scores.pkl")

print("\n✅ All Models Saved Successfully")
