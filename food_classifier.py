"""
DentiScan - Food Classifier
============================
Step 1 of the DentiScan pipeline: classify a food as Good / Moderation / Harmful
for dental health, based on its nutrient profile.

Data source (swap this in for real use):
  - Open Food Facts API (barcode lookups): https://world.openfoodfacts.org/data
  - USDA FoodData Central (whole foods):    https://fdc.nal.usda.gov/download-datasets

For this demo we use a small hand-built sample_food_data.csv with the same
columns you'd get by combining fields from those sources:
  food_name, category, sugar_g_100g, carbs_g_100g, fiber_g_100g, is_acidic, sticky_texture

Why label with rules first, then train a model?
-------------------------------------------------
There's no existing dataset that already says "this food is Good/Moderation/Harmful
for teeth" - that judgement doesn't exist as raw data anywhere. So we encode dental
domain knowledge as a labeling rule, apply it to the nutrient data to generate labels,
and then train a classifier on (nutrients -> label). The trained model is what you'd
actually ship in the app: it generalizes to foods it hasn't seen labeled explicitly,
and its feature importances make a good "why" explanation for the report.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def label_food(row: pd.Series) -> str:
    """Domain-knowledge rule for dental risk labeling.

    Sticky, high-sugar, acidic foods that also cling to teeth (raisins, candy,
    caramel) are worse than a high-sugar food that clears the mouth quickly
    (e.g. fruit juice). This rule is intentionally simple and explainable --
    for a report you can point a reader straight at the thresholds used.
    """
    sugar = row["sugar_g_100g"]
    acidic = row["is_acidic"]
    sticky = row["sticky_texture"]

    score = 0
    if sugar >= 30:
        score += 3
    elif sugar >= 10:
        score += 2
    elif sugar >= 3:
        score += 1

    if acidic:
        score += 1
    if sticky:
        score += 1

    if score >= 4:
        return "Harmful"
    elif score >= 2:
        return "Moderation"
    else:
        return "Good"


def build_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["label"] = df.apply(label_food, axis=1)
    return df


def train_model(df: pd.DataFrame):
    features = ["sugar_g_100g", "carbs_g_100g", "fiber_g_100g", "is_acidic", "sticky_texture"]
    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Food Classifier: Test Set Performance ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print("Classes:", list(model.classes_))
    print()
    print("=== Learned Decision Rules (for your report) ===")
    print(export_text(model, feature_names=features))

    return model, features


def predict_food(model, features, sugar_g_100g, carbs_g_100g, fiber_g_100g, is_acidic, sticky_texture):
    row = pd.DataFrame([{
        "sugar_g_100g": sugar_g_100g,
        "carbs_g_100g": carbs_g_100g,
        "fiber_g_100g": fiber_g_100g,
        "is_acidic": int(is_acidic),
        "sticky_texture": int(sticky_texture),
    }])[features]
    return model.predict(row)[0]


if __name__ == "__main__":
    df = build_dataset("sample_food_data.csv")
    print(df[["food_name", "sugar_g_100g", "is_acidic", "sticky_texture", "label"]].to_string(index=False))
    print()

    model, features = train_model(df)
    joblib.dump({"model": model, "features": features}, "food_classifier.joblib")

    # Example: classify a food not in the training set
    example = predict_food(model, features, sugar_g_100g=18, carbs_g_100g=20,
                            fiber_g_100g=0.5, is_acidic=1, sticky_texture=1)
    print(f"\nExample prediction (sugary, acidic, sticky snack) -> {example}")
