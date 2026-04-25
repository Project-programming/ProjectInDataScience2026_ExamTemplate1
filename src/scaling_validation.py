import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("features_validation.csv")

# Select only numeric columns automatically to avoid image object error
X_train = df.select_dtypes(include=["number"])

# Remove label column explicitly
X_train = X_train.drop(columns=["is_cancer"])



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

df_scaled = df.copy()
df_scaled[X_train.columns] = X_train_scaled

df_scaled.to_csv("features_validation_scaled.csv", index=False)
joblib.dump(scaler, "feature_scaler.pkl")
