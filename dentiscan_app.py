"""
DentiScan - Streamlit App (multi-page + profile wizard)
==========================================================
Run with:
    pip install streamlit pandas scikit-learn joblib numpy
    streamlit run dentiscan_app.py

Two separate screens, switched via the sidebar:
  1. "Your Profile"  -- one question per screen, Back/Next wizard
  2. "Check a Food"  -- search or manually enter a food, get the combined result

Loads the models trained by food_classifier.py and dental_risk_model.py
(the .joblib files already sit next to this script) and reuses the exact
recommendation logic from dentiscan_demo.py.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from food_classifier import predict_food, build_dataset, train_model as train_food_model
from dental_risk_model import predict_risk, train as train_dental_model, probability_to_tier
from dentiscan_demo import get_recommendation

APP_DIR = Path(__file__).parent
CATEGORY_ICON = {"Good": "🟢", "Moderation": "🟡", "Harmful": "🔴"}

# How much each food category nudges the session risk score, and how far
# that nudge is allowed to accumulate away from the profile's baseline.
RISK_NUDGE = {"Good": -0.015, "Moderation": 0.015, "Harmful": 0.04}
MAX_ADJUSTMENT = 0.30

st.set_page_config(page_title="DentiScan", page_icon="🦷", layout="centered")

# ---------------------------------------------------------------------------
# Profile questions -- one shown per screen in the wizard.
# `key` must match the argument names predict_risk() expects.
# ---------------------------------------------------------------------------
QUESTIONS = [
    {"key": "age", "label": "How old are you?", "type": "number",
     "min_value": 1, "max_value": 100, "default": 25},
    {"key": "gender", "label": "What's your gender?", "type": "select",
     "options": ["M", "F"]},
    {"key": "brushing_freq_per_day", "label": "How many times a day do you brush your teeth?",
     "type": "select", "options": [0, 1, 2, 3]},
    {"key": "flosses", "label": "Do you floss regularly?", "type": "yesno", "default": 0.0},
    {"key": "sugary_snacks_per_day", "label": "How many sugary snacks do you eat per day, on average?",
     "type": "slider", "min_value": 0, "max_value": 10, "default": 2},
    {"key": "sugary_drinks_per_day", "label": "How many sugary drinks (soda, juice, energy drinks) do you have per day?",
     "type": "slider", "min_value": 0, "max_value": 10, "default": 1},
    {"key": "smokes", "label": "Do you smoke?", "type": "yesno", "default": 0.0},
    {"key": "dental_visit_last_year", "label": "Have you visited a dentist in the last year?",
     "type": "yesno", "default": 1.0},
    {"key": "family_income_level", "label": "What's your household income level?",
     "type": "select", "options": ["low", "mid", "high"], "default_index": 1},
    {"key": "uses_fluoride_toothpaste", "label": "Do you use fluoride toothpaste?",
     "type": "yesno", "default": 1.0},
    {"key": "mouthwash_use", "label": "Do you use mouthwash regularly?", "type": "yesno", "default": 0.0},
    {"key": "teeth_grinding", "label": "Do you grind your teeth (bruxism), especially at night?",
     "type": "yesno", "default": 0.0},
    {"key": "chews_sugarfree_gum", "label": "Do you chew sugar-free gum after meals?",
     "type": "yesno", "default": 0.0},
]

YESNO_OPTIONS = ["No", "Sometimes", "Yes"]
YESNO_TO_VALUE = {"No": 0.0, "Sometimes": 0.5, "Yes": 1.0}
VALUE_TO_YESNO = {v: k for k, v in YESNO_TO_VALUE.items()}


@st.cache_resource
def load_models():
    """Train both models fresh on startup instead of loading pickled .joblib
    files -- this avoids numpy/scikit-learn version-mismatch errors between
    this sandbox and whatever environment the app is deployed on. Training
    on this small dataset takes well under a second."""
    food_df_local = build_dataset(str(APP_DIR / "sample_food_data.csv"))
    food_model, food_features = train_food_model(food_df_local)

    dental_pipeline = train_dental_model(str(APP_DIR / "dental_risk_data.csv"))

    return food_model, food_features, dental_pipeline


@st.cache_data
def load_food_data():
    return pd.read_csv(APP_DIR / "sample_food_data.csv")


food_model, food_features, dental_pipeline = load_models()
food_df = load_food_data()

if "profile" not in st.session_state:
    st.session_state.profile = None
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "step" not in st.session_state:
    st.session_state.step = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "risk_adjustment" not in st.session_state:
    st.session_state.risk_adjustment = 0.0

st.sidebar.title("🦷 DentiScan")
page = st.sidebar.radio("Go to", ["Your Profile", "Check a Food"])
if st.session_state.profile is not None:
    st.sidebar.success("Profile saved ✓")
else:
    st.sidebar.info("Profile not completed yet")


def render_question_widget(q: dict):
    key = q["key"]
    saved = st.session_state.answers.get(key, q.get("default"))
    widget_key = f"widget_{key}"

    if q["type"] == "number":
        return st.number_input(q["label"], min_value=q["min_value"], max_value=q["max_value"],
                                value=saved if saved is not None else q["min_value"], key=widget_key)
    if q["type"] == "select":
        options = q["options"]
        index = options.index(saved) if saved in options else q.get("default_index", 0)
        return st.selectbox(q["label"], options, index=index, key=widget_key)
    if q["type"] == "slider":
        return st.slider(q["label"], min_value=q["min_value"], max_value=q["max_value"],
                          value=saved if saved is not None else q["min_value"], key=widget_key)
    if q["type"] == "yesno":
        current_value = st.session_state.answers.get(key, q.get("default", 0.0))
        index = YESNO_OPTIONS.index(VALUE_TO_YESNO.get(current_value, "No"))
        choice = st.selectbox(q["label"], YESNO_OPTIONS, index=index, key=widget_key)
        return YESNO_TO_VALUE[choice]
    raise ValueError(f"Unknown question type: {q['type']}")


# ---------------------------------------------------------------------------
# PAGE 1: Your Profile (one question per screen)
# ---------------------------------------------------------------------------
if page == "Your Profile":
    st.title("Your Profile")

    total = len(QUESTIONS)
    step = st.session_state.step

    if step >= total:
        st.success("Profile complete!")

        baseline_risk = predict_risk(dental_pipeline, **st.session_state.profile)
        current_proba = max(0.0, min(0.99, baseline_risk["caries_probability"] + st.session_state.risk_adjustment))
        current_tier = probability_to_tier(current_proba)

        st.metric("Your current dental-caries risk", f"{current_tier} ({current_proba:.0%})")
        if st.session_state.risk_adjustment != 0:
            st.caption(f"Baseline from your profile: {baseline_risk['risk_tier']} ({baseline_risk['caries_probability']:.0%}) — adjusted by foods checked this session.")

        st.write("Head to **Check a Food** in the sidebar to see how a food affects this.")
        if st.button("Edit answers"):
            st.session_state.step = 0
            st.rerun()
    else:
        st.progress(step / total, text=f"Question {step + 1} of {total}")
        q = QUESTIONS[step]
        answer = render_question_widget(q)

        col_back, col_next = st.columns(2)
        with col_back:
            if step > 0 and st.button("⬅ Back", use_container_width=True):
                st.session_state.answers[q["key"]] = answer
                st.session_state.step -= 1
                st.rerun()
        with col_next:
            label = "Finish ✅" if step == total - 1 else "Next ➡"
            if st.button(label, use_container_width=True, type="primary"):
                st.session_state.answers[q["key"]] = answer
                st.session_state.step += 1
                if st.session_state.step >= total:
                    raw = st.session_state.answers
                    st.session_state.profile = dict(
                        age=int(raw["age"]),
                        gender=raw["gender"],
                        brushing_freq_per_day=int(raw["brushing_freq_per_day"]),
                        flosses=float(raw["flosses"]),
                        sugary_snacks_per_day=int(raw["sugary_snacks_per_day"]),
                        sugary_drinks_per_day=int(raw["sugary_drinks_per_day"]),
                        smokes=float(raw["smokes"]),
                        dental_visit_last_year=float(raw["dental_visit_last_year"]),
                        family_income_level=raw["family_income_level"],
                        uses_fluoride_toothpaste=float(raw["uses_fluoride_toothpaste"]),
                        mouthwash_use=float(raw["mouthwash_use"]),
                        teeth_grinding=float(raw["teeth_grinding"]),
                        chews_sugarfree_gum=float(raw["chews_sugarfree_gum"]),
                    )
                    st.session_state.risk_adjustment = 0.0
                st.rerun()

# ---------------------------------------------------------------------------
# PAGE 2: Check a Food
# ---------------------------------------------------------------------------
else:
    st.title("Check a Food")

    if st.session_state.profile is None:
        st.warning("Complete **Your Profile** first (see sidebar) to get a personalized risk result.")
        st.stop()

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

    if nutrients is not None:
        category = predict_food(food_model, food_features, **nutrients)
        risk = predict_risk(dental_pipeline, **st.session_state.profile)

        # Nudge the session risk score based on this food, then clamp it
        # within MAX_ADJUSTMENT of the profile's fixed baseline.
        st.session_state.risk_adjustment = max(
            -MAX_ADJUSTMENT,
            min(MAX_ADJUSTMENT, st.session_state.risk_adjustment + RISK_NUDGE[category]),
        )
        adjusted_proba = max(0.0, min(0.99, risk["caries_probability"] + st.session_state.risk_adjustment))
        adjusted_tier = probability_to_tier(adjusted_proba)

        message = get_recommendation(category, adjusted_tier)

        st.divider()
        st.subheader(f"{CATEGORY_ICON[category]} {food_label}")
        col1, col2 = st.columns(2)
        col1.metric("Food category", category)
        col2.metric(
            "Your current caries risk",
            f"{adjusted_tier} ({adjusted_proba:.0%})",
            delta=f"{st.session_state.risk_adjustment:+.0%} from baseline",
            delta_color="inverse",
        )
        st.caption(f"Baseline risk from your profile: {risk['risk_tier']} ({risk['caries_probability']:.0%})")
        st.write(message)

        st.session_state.history.append(
            {
                "Food": food_label,
                "Category": category,
                "Risk at the time": f"{adjusted_tier} ({adjusted_proba:.0%})",
                "Recommendation": message,
            }
        )

    if st.session_state.history:
        st.divider()
        st.subheader("Session history")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Clear history & reset session risk"):
            st.session_state.history = []
            st.session_state.risk_adjustment = 0.0
            st.rerun()
