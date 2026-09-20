"""
DentiScan - Streamlit App
============================
Run with:
    pip install streamlit pandas scikit-learn joblib numpy
    streamlit run dentiscan_app.py

Loads the models trained by food_classifier.py and dental_risk_model.py
(the .joblib files already sit next to this script) and reuses the exact
recommendation logic from dentiscan_demo.py.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from food_classifier import predict_food
from dental_risk_model import predict_risk
from dentiscan_demo import get_recommendation

APP_DIR = Path(__file__).parent
CATEGORY_ICON = {"Good": "🟢", "Moderation": "🟡", "Harmful": "🔴"}

st.set_page_config(page_title="DentiScan", page_icon="🦷", layout="centered")


@st.cache_resource
def load_models():
    food_bundle = joblib.load(APP_DIR / "food_classifier.joblib")
    dental_pipeline = joblib.load(APP_DIR / "dental_risk_model.joblib")
    return food_bundle["model"], food_bundle["features"], dental_pipeline


@st.cache_data
def load_food_data():
    return pd.read_csv(APP_DIR / "sample_food_data.csv")


food_model, food_features, dental_pipeline = load_models()
food_df = load_food_data()

if "profile" not in st.session_state:
    st.session_state.profile = None
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🦷 DentiScan")
st.caption("Search or enter a food to see how it affects your dental health, personalized to you.")

# ---------------------------------------------------------------------------
# Step 1: user profile (collected once, reused for every food check)
# ---------------------------------------------------------------------------
with st.expander("Your profile", expanded=st.session_state.profile is None):
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=100, value=25)
            gender = st.selectbox("Gender", ["M", "F"])
            brushing = st.selectbox("Brushing frequency per day", [0, 1, 2, 3], index=2)
            flosses = st.checkbox("Flosses regularly")
        with col2:
            sugary_snacks = st.slider("Sugary snacks per day", 0, 10, 2)
            smokes = st.checkbox("Smokes")
            dental_visit = st.checkbox("Dental visit in the last year", value=True)
            income = st.selectbox("Household income level", ["low", "mid", "high"], index=1)

        if st.form_submit_button("Save profile"):
            st.session_state.profile = dict(
                age=age,
                gender=gender,
                brushing_freq_per_day=brushing,
                flosses=int(flosses),
                sugary_snacks_per_day=sugary_snacks,
                smokes=int(smokes),
                dental_visit_last_year=int(dental_visit),
                family_income_level=income,
            )
            st.success("Profile saved.")
            st.rerun()

if st.session_state.profile is None:
    st.info("Fill in your profile above to get personalized recommendations.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Step 2: check a food (from the sample list, or enter nutrients manually --
# this second tab is where a real barcode/Open Food Facts lookup would feed in)
# ---------------------------------------------------------------------------
st.subheader("Check a food")
tab_search, tab_manual = st.tabs(["Search food list", "Enter nutrients manually"])

nutrients = None
food_label = None

with tab_search:
    food_name = st.selectbox("Choose a food", sorted(food_df["food_name"].tolist()))
    if st.button("Check this food", key="check_list"):
        row = food_df.loc[food_df["food_name"] == food_name].iloc[0]
        nutrients = dict(
            sugar_g_100g=float(row["sugar_g_100g"]),
            carbs_g_100g=float(row["carbs_g_100g"]),
            fiber_g_100g=float(row["fiber_g_100g"]),
            is_acidic=int(row["is_acidic"]),
            sticky_texture=int(row["sticky_texture"]),
        )
        food_label = food_name

with tab_manual:
    st.caption("Use this for a food not in the sample list, e.g. pulled from a barcode API.")
    manual_name = st.text_input("Food name", value="Custom food")
    sugar = st.number_input("Sugar (g per 100g)", min_value=0.0, value=10.0)
    carbs = st.number_input("Carbs (g per 100g)", min_value=0.0, value=15.0)
    fiber = st.number_input("Fiber (g per 100g)", min_value=0.0, value=1.0)
    acidic = st.checkbox("Acidic (citrus, soda, vinegar-based)")
    sticky = st.checkbox("Sticky / clings to teeth")
    if st.button("Check custom food", key="check_manual"):
        nutrients = dict(
            sugar_g_100g=sugar,
            carbs_g_100g=carbs,
            fiber_g_100g=fiber,
            is_acidic=int(acidic),
            sticky_texture=int(sticky),
        )
        food_label = manual_name or "Custom food"

# ---------------------------------------------------------------------------
# Step 3: combine both models into one result
# ---------------------------------------------------------------------------
if nutrients is not None:
    category = predict_food(food_model, food_features, **nutrients)
    risk = predict_risk(dental_pipeline, **st.session_state.profile)
    message = get_recommendation(category, risk["risk_tier"])

    st.divider()
    st.subheader(f"{CATEGORY_ICON[category]} {food_label}")
    col1, col2 = st.columns(2)
    col1.metric("Food category", category)
    col2.metric("Your caries risk", f"{risk['risk_tier']} ({risk['caries_probability']:.0%})")
    st.write(message)

    st.session_state.history.append(
        {
            "Food": food_label,
            "Category": category,
            "Your risk tier": risk["risk_tier"],
            "Recommendation": message,
        }
    )

# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader("Session history")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
