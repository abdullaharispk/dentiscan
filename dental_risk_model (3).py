"""
DentiScan - Dental Risk Model
================================
Step 2 of the DentiScan pipeline: predict a user's dental-caries risk from
demographic, lifestyle, and diet-adjacent factors (this is the "considers
the user's information" step in the project description).

Swap the synthetic dental_risk_data.csv (see generate_dental_risk_data.py)
for a cleaned NHANES extract or the Hugging Face oral-health dataset once
you've done that preprocessing -- keep the same column names and this script
still works.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

NUMERIC_FEATURES = [
    "age", "brushing_freq_per_day", "flosses",
    "sugary_snacks_per_day", "sugary_drinks_per_day", "smokes",
    "dental_visit_last_year", "uses_fluoride_toothpaste",
    "mouthwash_use", "teeth_grinding", "chews_sugarfree_gum",
]
CATEGORICAL_FEATURES = ["gender", "family_income_level"]
TARGET = "has_caries"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model),
    ])


def train(csv_path: str = "dental_risk_data.csv"):
    df = pd.read_csv(csv_path)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=== Dental Risk Model: Test Set Performance ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # Feature importance (approximate, after one-hot expansion)
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = cat_names + NUMERIC_FEATURES
    importances = pipeline.named_steps["classifier"].feature_importances_
    ranked = sorted(zip(all_names, importances), key=lambda x: -x[1])
    print("\n=== Feature Importances (for your report) ===")
    for name, imp in ranked:
        print(f"  {name:30s} {imp:.3f}")

    return pipeline


def predict_risk(pipeline, **kwargs) -> dict:
    """kwargs must include: age, gender, brushing_freq_per_day, flosses,
    sugary_snacks_per_day, smokes, dental_visit_last_year, family_income_level"""
    row = pd.DataFrame([kwargs])[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    proba = pipeline.predict_proba(row)[0, 1]
    tier = "High" if proba >= 0.6 else "Moderate" if proba >= 0.3 else "Low"
    return {"caries_probability": round(float(proba), 3), "risk_tier": tier}


if __name__ == "__main__":
    pipeline = train()
    joblib.dump(pipeline, "dental_risk_model.joblib")

    example = predict_risk(
        pipeline,
        age=24, gender="F", brushing_freq_per_day=1, flosses=0,
        sugary_snacks_per_day=4, sugary_drinks_per_day=2, smokes=0,
        dental_visit_last_year=0, family_income_level="low",
        uses_fluoride_toothpaste=1, mouthwash_use=0,
        teeth_grinding=0, chews_sugarfree_gum=0,
    )
    print(f"\nExample user risk prediction: {example}")
