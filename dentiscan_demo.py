"""
DentiScan - End-to-End Demo
==============================
Ties the two models together the way the real app would:

  User scans/searches a food  ---->  food_classifier.py  ----> Good / Moderation / Harmful
  User profile (age, habits)  ---->  dental_risk_model.py ----> Low / Moderate / High risk
                                              |
                                              v
                                 decision table -> final personalized message

Run food_classifier.py and dental_risk_model.py once first to produce the
.joblib model files this script loads.
"""

import joblib
from food_classifier import predict_food
from dental_risk_model import predict_risk


RECOMMENDATION_TABLE = {
    ("Good", "Low"): "Great choice - safe for your teeth, no special precautions needed.",
    ("Good", "Moderate"): "Good food choice. Keep up regular brushing since your risk is moderate.",
    ("Good", "High"): "Good food choice, but with your current risk level, don't skip your daily brushing/flossing.",
    ("Moderation", "Low"): "Fine in moderation. Try not to make this a daily habit.",
    ("Moderation", "Moderate"): "Limit how often you have this, and rinse with water afterward.",
    ("Moderation", "High"): "Given your risk level, treat this as an occasional food only - rinse or brush after eating it.",
    ("Harmful", "Low"): "High sugar/acid content. Rinse your mouth with water afterward.",
    ("Harmful", "Moderate"): "This food is hard on teeth. Limit frequency and rinse/brush after eating.",
    ("Harmful", "High"): "This food is high-risk for your teeth, and your personal risk is already high - avoid frequent consumption and see a dentist regularly.",
}


def get_recommendation(food_category: str, risk_tier: str) -> str:
    return RECOMMENDATION_TABLE[(food_category, risk_tier)]


def run_demo():
    food_bundle = joblib.load("food_classifier.joblib")
    food_model, food_features = food_bundle["model"], food_bundle["features"]
    dental_pipeline = joblib.load("dental_risk_model.joblib")

    # --- Example: a user scans a chocolate bar ---
    food_category = predict_food(
        food_model, food_features,
        sugar_g_100g=51.5, carbs_g_100g=59.4, fiber_g_100g=3.4,
        is_acidic=0, sticky_texture=1,
    )

    risk_result = predict_risk(
        dental_pipeline,
        age=17, gender="M", brushing_freq_per_day=1, flosses=0,
        sugary_snacks_per_day=3, sugary_drinks_per_day=2, smokes=0,
        dental_visit_last_year=0, family_income_level="mid",
        uses_fluoride_toothpaste=1, mouthwash_use=0,
        teeth_grinding=0, chews_sugarfree_gum=0,
    )

    message = get_recommendation(food_category, risk_result["risk_tier"])

    print("Scanned food: Milk Chocolate")
    print(f"  -> Food category: {food_category}")
    print(f"User dental risk: {risk_result}")
    print(f"\nDentiScan says: {message}")


if __name__ == "__main__":
    run_demo()
