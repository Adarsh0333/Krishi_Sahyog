import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# 1. Load Dataset
df = pd.read_csv("dataset/crop_recommendation.csv")
feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[feature_cols]
y = df["label"]

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Benchmark Candidate Models with 5-Fold Cross-Validation
candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_name = None
best_cv_score = -1.0
best_model = None

print("--- Model Benchmarking (5-Fold CV Accuracy) ---")
for name, clf in candidates.items():
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    mean_acc = scores.mean()
    print(f"{name}: {mean_acc * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")
    if mean_acc > best_cv_score:
        best_cv_score = mean_acc
        best_name = name
        best_model = clf

# 4. Fit Best Model on Full Training Set & Evaluate on Holdout Test Set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="macro")

print(f"\nSelected Model: {best_name}")
print(f"Holdout Test Accuracy: {test_acc * 100:.2f}% | Macro F1-Score: {test_f1 * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Extract Ideal Agronomic Profiles per Crop (25th-75th percentiles & medians)
crop_profiles = {}
for crop, group in df.groupby("label"):
    crop_profiles[crop] = {
        col: {
            "min": round(float(group[col].quantile(0.15)), 1),
            "ideal": round(float(group[col].median()), 1),
            "max": round(float(group[col].quantile(0.85)), 1),
        }
        for col in feature_cols
    }

# 6. Save Unified Model Artifact
artifact = {
    "model": best_model,
    "model_name": best_name,
    "feature_cols": feature_cols,
    "test_accuracy": round(float(test_acc * 100), 2),
    "macro_f1": round(float(test_f1 * 100), 2),
    "crop_profiles": crop_profiles,
}

joblib.dump(artifact, "crop_model.pkl")
print("Saved upgraded model bundle to crop_model.pkl!")
