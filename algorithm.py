import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# load csv
df = pd.read_csv("Exoplanet_Dataset_Cleaned_Filtered.csv")

# Drop identifier column
X = df.drop(columns=["pl_name", "pl_hab"])
y = df["pl_hab"]


# class imbalance
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"Class balance:\n{y.value_counts(normalize=True)}")
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# Classifing model
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



# training and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#train model
xgb.fit(X_train, y_train)

#classification report
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

#example prediction
sample = X_test.iloc[[0]]
true_label = y_test.iloc[0]

pred = xgb.predict(sample)[0]
proba = xgb.predict_proba(sample)[0]

print("\nExample Prediction:")
print(f"True label: {true_label}")
print(f"Predicted class: {pred}")
print(f"Confidence (probability): {proba}")

# user input
print("\n--- Custom Planet Prediction ---")
user_data = {}

for col in X.columns:
    dtype = X[col].dtype
    val = input(f"Enter value for {col} (leave blank for missing): ")
    if val == "":
        user_data[col] = None
    else:
        # Cast to numeric if possible
        if dtype == "float64" or dtype == "int64":
            try:
                user_data[col] = float(val)
            except ValueError:
                user_data[col] = None
        else:
            user_data[col] = val

# Convert to DataFrame
user_df = pd.DataFrame([user_data])

# Predict
user_pred = xgb.predict(user_df)[0]
user_proba = xgb.predict_proba(user_df)[0]

print("\nPrediction for your custom planet:")
print(f"Predicted class: {user_pred} (0 = Not Habitable, 1 = Potentially Habitable)")
print(f"Confidence (probability): {user_proba}")
