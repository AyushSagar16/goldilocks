import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier, plot_tree
import matplotlib.pyplot as plt


# =============================
# 1. Load Data
# =============================
df = pd.read_csv("Exoplanet_Dataset_Cleaned_Filtered.csv")

X = df.drop(columns=["pl_name", "pl_hab"])
y = df["pl_hab"]

numeric_features = X.columns.tolist()



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





X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
xgb.fit(X_train, y_train)

# =============================
# 2. Streamlit UI
# =============================
st.title("Goldilocks: Exoplanet Habitability Classifier")
st.write("Adjust the sliders to input planet/star properties.")

user_data = {}

user_data["pl_orbper"] = st.slider("Orbital Period (days)", 0.1, 1000.0, 365.0) # Changed to a reasonable max
user_data["pl_rade"] = st.slider("Planet Radius (Earth radii)", 0.1, 20.0, 1.0)
user_data["pl_masse"] = st.slider("Planet Mass (Earth masses)", 0.1, 100.0, 1.0)
user_data["pl_eqt"] = st.slider("Equilibrium Temperature (K)", 100.0, 3000.0, 288.0)
user_data["st_teff"] = st.slider("Star Temperature (K)", 2000, 10000, 5800)
user_data["sy_dist"] = st.slider("System Distance (parsecs)", 0.1, 1000.0, 10.0)

# Convert to DataFrame
user_df = pd.DataFrame([user_data])

if st.button("Predict Habitability"):
    pred = xgb.predict(user_df)[0]
    proba = xgb.predict_proba(user_df)[0]

    st.subheader("Prediction Result:")
    if pred == 1:
        st.success(f"🌱 Potentially Habitable (Confidence: {proba[1]*100:.2f}%)")
    else:
        st.error(f"💀 Not Habitable (Confidence: {proba[0]*100:.2f}%)")

    # st.write("Class probabilities:", proba)

    # =============================
    # 3. Horizontal Probability Bar Chart
    # =============================
    labels = ["Not Habitable (0)", "Potentially Habitable (1)"]
    colors = ["red", "green"]

    fig, ax = plt.subplots(figsize=(6, 2))
    bars = ax.barh(labels, proba, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Probability Distribution")

    # Add text labels on bars
    for bar, p in zip(bars, proba):
        ax.text(p + 0.01, bar.get_y() + bar.get_height()/2,
                f"{p*100:.2f}%", va="center")


