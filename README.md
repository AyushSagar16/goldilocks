# 🪐 Goldilocks: Exoplanet Habitability Classifier  

Goldilocks is a machine learning project that predicts whether an exoplanet is **potentially habitable** based on its physical and stellar properties.  
The project is built with **XGBoost** and deployed as an interactive **Streamlit web app**.  

---

## 🚀 Features  

- **Interactive UI** with sliders to input exoplanet properties.  
- **Prediction output**: Determines if a planet is habitable (`1`) or not habitable (`0`) with probability confidence.  
- **Class imbalance handling** using `scale_pos_weight` in XGBoost (habitable planets are rare!).  
- **Model Insights** tab with visualization of decision trees.  
- **Validation tab** including:  
  - ROC Curve with AUC score  
  - Confusion Matrix  

---

## 🧪 Data  

- Dataset: `Exoplanet_Dataset_Cleaned_Filtered.csv`  
- Features include:  
  - **Orbital period** (days)  
  - **Planet radius** (Earth radii)  
  - **Planet mass** (Earth masses)  
  - **Equilibrium temperature** (Kelvin)  
  - **Star temperature** (Kelvin)  
  - **System distance** (parsecs)  

Target column:  
- `pl_hab` → `1` if potentially habitable, `0` otherwise  

---

## 🧑‍💻 Tech Stack  

- **Python**  
- **Streamlit** – interactive frontend  
- **XGBoost** – classification algorithm  
- **Scikit-learn** – model evaluation  
- **Matplotlib** – visualizations  

---

## ⚙️ Installation & Usage  

1. Clone the repository:  
   ```bash
   git clone https://github.com/AyushSagar16/goldilocks.git
   cd goldilocks
