import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load Dataset
# Expected columns: N, P, K, temperature, humidity, ph, rainfall, label (crop)
df = pd.read_csv("dataset/crop_recommendation.csv")

# 2. Define Features (X) and Target (y)
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# 3. Split and Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save Model Artifact
joblib.dump(model, "crop_model.pkl")
print("Model trained successfully and saved as crop_model.pkl!")