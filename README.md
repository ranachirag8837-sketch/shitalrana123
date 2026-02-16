# Student Result Prediction System (ML + Streamlit)

A Machine Learning based web application that predicts student exam marks using study hours, attendance, internal marks, and assignment score.

The system compares multiple ML models, selects the best-performing algorithm automatically, and also shows a hybrid ensemble prediction.

---

## Features

- Predict student marks using trained ML models  
- Compare multiple algorithms  
- Automatically detect BEST performing model  
- Hybrid Ensemble Prediction (Top models average)  
- Attendance-based personalized feedback  
- Interactive Streamlit dashboard UI  
- Graph comparison of model outputs  

---

## Machine Learning Models Used

- Linear Regression  
- Random Forest  
- Support Vector Regressor  
- Other trained models (stored in models.pkl)

Best model is automatically selected based on highest accuracy.

---

## Project Structure

C:.
+---.idea
|   |   .gitignore
|   |   misc.xml
|   |   modules.xml
|   |   Rana Chirag.iml
|   |   workspace.xml
|   |
|   \---inspectionProfiles
|           profiles_settings.xml
|           Project_Default.xml
|
+---App
|       app.py
|
+---Dataset
|       student_data.csv
|
\---Model
        model.py
        models.pkl
        scaler.pkl
        scores.pkl
        top_models.pkl
