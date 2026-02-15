import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


st.title("Machine Learning Classification Models")

# Upload dataset
uploaded_file = st.file_uploader("metrics.csv", type=["csv"])

model_option = st.selectbox(
    "Select Model",
    (
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    )
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    model = joblib.load(f"models/{model_option}.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    model_features = joblib.load("models/features.pkl")
   

    target_column = df.columns[-1]

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # ✅ Convert to Binary (MATCH TRAINING)
    y = (y > 0).astype(int)

    # Encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Feature Alignment
    X = X.reindex(columns=model_features, fill_value=0)

    # Missing value handling
    X = imputer.transform(X)

    # Scaling only for required models
    if model_option in ["logistic_regression", "knn"]:
        X = scaler.transform(X)
    

    y_pred = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)
