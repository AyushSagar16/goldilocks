import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =============================
# 1. Load Data & Train Model
# =============================
df = pd.read_csv("Exoplanet_Dataset_Cleaned.csv")

X = df.drop(columns=["pl_name", "pl_hab"])
y = df["pl_hab"]

categorical = X.select_dtypes(include=["object"]).columns
numeric = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), numeric),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ]
)

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False
)

clf = Pipeline(steps=[("preprocessor", preprocessor),
                     ("classifier", xgb)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
clf.fit(X_train, y_train)

# =============================
# 2. Streamlit UI
# =============================
st.title("🌍 Exoplanet Habitability Classifier")
st.write("Enter planet and star properties to predict habitability.")

# Create input widgets for numeric + categorical
user_data = {}
for col in X.columns:
    if col in numeric:
        user_data[col] = st.number_input(f"{col}", value=0.0, format="%.5f")
    else:
        user_data[col] = st.text_input(f"{col}", "")

# Prediction button
if st.button("Predict Habitability"):
    user_df = pd.DataFrame([user_data])
    pred = clf.predict(user_df)[0]
    proba = clf.predict_proba(user_df)[0]

    st.subheader("Prediction Result:")
    if pred == 1:
        st.success(f"🌱 Potentially Habitable (Confidence: {proba[1]*100:.2f}%)")
    else:
        st.error(f"💀 Not Habitable (Confidence: {proba[0]*100:.2f}%)")

    st.write("Class probabilities:", proba)
    