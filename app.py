import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Loan Prediction Dashboard", layout="wide")

# --- Helper functions ---
@st.cache_data
def load_data(path="train.csv.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_df(df):
    df = df.copy()
    # drop ID
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
    # Clean Dependents column (convert '3+' to 3)
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", "3")
    # Map target if present
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y":1, "N":0})
    return df

@st.cache_data
def build_and_train(df):
    df = preprocess_df(df)
    # Separate X and y
    if "Loan_Status" not in df.columns:
        raise ValueError("Dataset must contain 'Loan_Status' column as target.")
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"].astype(int)
    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    # transformers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return clf, acc, cm, report, X_test, y_test

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_target_distribution(df):
    fig, ax = plt.subplots()
    if "Loan_Status" in df.columns:
        counts = df["Loan_Status"].map({1:"Y",0:"N"}).value_counts()
        counts.plot.bar(ax=ax)
        ax.set_title("Loan Status Distribution (Y / N)")
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "No target column 'Loan_Status' found.", ha="center")
    return fig

# --- UI ---
st.markdown("<h1 style='color:#0b3d91'>Bank Loan Prediction — Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")
col1, col2 = st.columns([1,2])

with col1:
    st.sidebar.header("Controls")
    uploaded = st.sidebar.file_uploader("Upload CSV file (optional)", type=["csv"])
    st.sidebar.markdown("**Model:** Decision Tree Classifier (trained on dataset)")
    st.sidebar.markdown("**Theme:** Professional — blue & white")
    st.sidebar.caption("Tip: Upload a CSV with the same schema as the provided dataset to run predictions.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Predict")
    with st.sidebar.form("predict_form"):
        # dynamic form based on example columns if available
        df_example = load_data() if uploaded is None else pd.read_csv(uploaded)
        df_example = preprocess_df(df_example.copy())
        # Remove target if present
        if "Loan_Status" in df_example.columns:
            df_example = df_example.drop(columns=["Loan_Status"])
        inputs = {}
        for col in df_example.columns:
            if df_example[col].dtype == "object":
                opts = df_example[col].dropna().unique().tolist()[:10]
                inputs[col] = st.selectbox(col, options=opts, key=f"sb_{col}")
            else:
                val = float(df_example[col].median(skipna=True)) if pd.api.types.is_numeric_dtype(df_example[col]) else 0.0
                inputs[col] = st.number_input(col, value=val, key=f"num_{col}")
        submitted = st.form_submit_button("Predict")

with col2:
    # Load data
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("Uploaded CSV loaded.")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            df = load_data()
    else:
        df = load_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(50))

    # EDA
    st.subheader("Exploratory Data Analysis")
    st.markdown("**Target distribution**")
    fig1 = plot_target_distribution(preprocess_df(df.copy()))
    st.pyplot(fig1)

    st.markdown("**Sample feature correlations (numeric)**")
    numeric = preprocess_df(df.copy()).select_dtypes(include=[np.number])
    if not numeric.empty:
        fig2, ax = plt.subplots()
        sns.heatmap(numeric.corr(), annot=True, fmt='.2f', ax=ax)
        st.pyplot(fig2)
    else:
        st.write("No numeric columns to show correlation.")

    # Train model
    st.subheader("Train Decision Tree (on current dataset)")
    with st.spinner("Training model..."):
        try:
            model, acc, cm, report, X_test, y_test = build_and_train(df.copy())
            st.success(f"Model trained. Test accuracy: {acc:.3f}")
            st.markdown("**Confusion Matrix**")
            st.pyplot(plot_confusion_matrix(cm))
            st.markdown("**Classification Report (precision / recall / f1-score)**")
            st.table(pd.DataFrame(report).transpose())
        except Exception as e:
            st.error(f"Training failed: {e}")

    # Predict from quick form
    st.subheader("Manual Prediction")
    if 'submitted' in locals() and submitted:
        try:
            input_df = pd.DataFrame([inputs])
            preds = model.predict(input_df)
            proba = model.predict_proba(input_df)[:,1] if hasattr(model, "predict_proba") else None
            label = "Y" if preds[0]==1 else "N"
            st.write(f"Predicted Loan Status: **{label}**")
            if proba is not None:
                st.write(f"Confidence (probability of Y): {proba[0]:.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Batch predict for uploaded file
    st.subheader("Batch Predictions (optional)")
    if uploaded is not None:
        if "Loan_Status" in df.columns:
            st.info("Uploaded file already contains 'Loan_Status'. Predictions will be shown side-by-side.")
        try:
            X = preprocess_df(df.copy()).drop(columns=["Loan_Status"]) if "Loan_Status" in df.columns else preprocess_df(df.copy())
            preds = model.predict(X)
            df_result = df.copy()
            df_result["Predicted_Loan_Status"] = np.where(preds==1, "Y", "N")
            st.dataframe(df_result.head(100))
            # Offer download
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.markdown("---")
st.markdown("Built with ❤️ — Theme: Professional (blue).")
