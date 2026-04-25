import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("features_testing.csv")

# Select only numeric columns automatically to avoid image object error
X_train = df.select_dtypes(include=["number"])

# Remove label column explicitly
X_test = X_train.drop(columns=["is_cancer"])



scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

df_scaled = df.copy()
df_scaled[X_test.columns] = X_test_scaled

df_scaled.to_csv("features_test_scaled.csv", index=False)
joblib.dump(scaler, "feature_scaler.pkl")
