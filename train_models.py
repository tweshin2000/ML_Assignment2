import pandas as pd
import numpy as np
import joblib


import os
os.makedirs("models", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("heart_disease_uci.csv")

df = pd.get_dummies(df, drop_first=True)

target_column = df.columns[-1]

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert to Binary Classification
y = (y > 0).astype(int)

# SAVE FEATURE NAMES
joblib.dump(X.columns.tolist(), "models/features.pkl")

# Handle Missing Values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

joblib.dump(imputer, "models/imputer.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")


models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric='logloss')
}

metrics = {}

for name, model in models.items():

    # ✅ Use scaled data only for sensitive models
    if name in ["logistic_regression", "knn"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    metrics[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    joblib.dump(model, f"models/{name}.pkl")

metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv("metrics.csv")

print(metrics_df)
