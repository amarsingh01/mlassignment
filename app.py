import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# --- Page Config ---
st.set_page_config(page_title="Dynamic Loan Classifier", layout="wide")

# --- Cache the Training Process ---
# We cache this so the model doesn't re-train every time you click a button
@st.cache_resource
def train_selected_model(model_name):
    try:
        train_df = pd.read_csv('loan_data_train.csv')
    except FileNotFoundError:
        st.error("Error: 'loan_data_train.csv' not found. Please run the split script first.")
        return None, None

    cat_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    num_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    
    X_train = train_df.drop('loan_status', axis=1)
    y_train = train_df['loan_status']

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
    
    X_train_proc = preprocessor.fit_transform(X_train)

    # Model Selection
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "XGBoost":
        if xgb:
            model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        else:
            st.error("XGBoost library not found. Falling back to Random Forest.")
            model = RandomForestClassifier(random_state=42)
    
    # Training model
    model.fit(X_train_proc, y_train)
    return model, preprocessor

st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload 'loan_data_test.csv'", type=["csv"])
model_option = st.sidebar.selectbox("Select Model to Trigger", 
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

# --- Main App Logic ---
st.title("🏦 Live Model Inference Dashboard")

if uploaded_file is not None:
    # Trigger training/loading
    with st.spinner(f"Training {model_option}..."):
        model, preprocessor = train_selected_model(model_option)
    
    if model:
        # Load test data
        test_df = pd.read_csv(uploaded_file)
        X_test = test_df.drop('loan_status', axis=1)
        y_test = test_df['loan_status']

        # Process test data
        X_test_proc = preprocessor.transform(X_test)

        # Run Predictions (DYNAMIC)
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1]

        # Calculate Metrics (DYNAMIC)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # UI: Display Metrics
        st.subheader(f"📊 Live Metrics for {model_option}")
        cols = st.columns(6)
        cols[0].metric("Accuracy", f"{acc:.4f}")
        cols[1].metric("AUC Score", f"{auc:.4f}")
        cols[2].metric("Precision", f"{prec:.4f}")
        cols[3].metric("Recall", f"{rec:.4f}")
        cols[4].metric("F1 Score", f"{f1:.4f}")
        cols[5].metric("MCC", f"{mcc:.4f}")

        # UI: Confusion Matrix
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
        
        with c2:
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

else:
    st.info("👈 Upload 'loan_data_test.csv' in the sidebar to trigger the models.")