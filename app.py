import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier, plot_tree

# streamlit title and icon
st.set_page_config(page_title="Goldilocks: Exoplanet Habitability", page_icon="🪐", layout="wide")

# load data
df = pd.read_csv("Exoplanet_Dataset_Cleaned_Filtered.csv")
X = df.drop(columns=["pl_name", "pl_hab"])
y = df["pl_hab"].astype(int)
numeric_features = X.columns.tolist()

# Handle class imbalance
scale_pos_weight = (y == 0).sum() / max(1, (y == 1).sum())

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)

def fit_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    xgb.fit(X_train, y_train)
    return xgb, X_train, X_test, y_train, y_test

xgb, X_train, X_test, y_train, y_test = fit_model()

# sdidebar info
st.sidebar.header("📖 Description of Model")
st.sidebar.markdown(
    """
    This app uses an **XGBoost classifier** trained on exoplanet and stellar properties 
    to predict whether a planet is **potentially habitable (1)** or **not habitable (0)**.

    - **Algorithm**: Gradient Boosted Decision Trees (XGBoost)  
    - **Features**: Orbital period, radius, mass, equilibrium temperature, stellar temperature, distance  
    - **Handling imbalance**: Weights are applied so the rare habitable planets count more.  
    - **Outputs**: Predicted class and probability confidence.  

    You can adjust the sliders to simulate exoplanet characteristics and see the model’s prediction.
    """
)

# title and sliders
st.title("🪐 Goldilocks: Exoplanet Habitability Classifier")
st.caption("Move the sliders, hit **Predict**, and explore model insights.")

c1, c2, c3 = st.columns(3)
with c1:
    pl_orbper = st.slider("Orbital Period (days)", 0.1, 1000.0, 365.0) # Changed to a reasonable max
    pl_rade   = st.slider("Planet Radius (Earth radii)", 0.1, 25.0, 1.0) # chagned this too
with c2:
    pl_masse  = st.slider("Planet Mass (Earth masses)", 0.1, 100.0, 1.0)
    pl_eqt    = st.slider("Equilibrium Temp (K)", 100.0, 1000.0, 288.0) #changed max
with c3:
    st_teff   = st.slider("Star Temperature (K)", 2000, 10000, 5800)
    sy_dist   = st.slider("System Distance (pc)", 0.1, 1000.0, 10.0)

user_df = pd.DataFrame([{
    "pl_orbper": pl_orbper,
    "pl_rade": pl_rade,
    "pl_masse": pl_masse,
    "pl_eqt": pl_eqt,
    "st_teff": st_teff,
    "sy_dist": sy_dist,
}])

# prediction button
if st.button("Predict Habitability"):
    proba = xgb.predict_proba(user_df)[0]
    p1 = float(proba[1])
    pred = int(xgb.predict(user_df)[0])

    
    
    if pred == 1:
        st.success(f"**Potentially Habitable**\n\nConfidence: **{p1*100:.2f}%**")
    else:
        st.error(f"**Not Habitable**\n\nConfidence: **{(1-p1)*100:.2f}%**")
    st.progress(p1)

# Tabs
tab1, tab2 = st.tabs(["📈 Model Insights", "🧪 Validation"])

with tab1:
    st.subheader("Example Decision Tree")
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(xgb, num_trees=0, ax=ax)
    st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("ROC Curve & Confusion Matrix")
    y_proba = xgb.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0,1],[0,1], lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend(loc="lower right")
        st.pyplot(fig, width='stretch')
    with c2:
        y_pred_thr = xgb.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_thr, labels=[0,1])
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, width='stretch')
    