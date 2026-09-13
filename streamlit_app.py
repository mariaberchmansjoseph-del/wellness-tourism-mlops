import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── PAGE CONFIGURATION ────────────────────────────────────────
st.set_page_config(
    page_title = "Wellness Tourism Predictor",
    page_icon  = "✈️",
    layout     = "wide"
)

# ── LOAD MODEL AND SUPPORTING FILES ───────────────────────────
@st.cache_resource
def load_model():
    """Load the best model from the repository."""
    path = "models/best_model.pkl"
    if not os.path.exists(path):
        st.error("Model file not found. Please run the training pipeline.")
        return None
    return joblib.load(path)

@st.cache_data
def load_metadata():
    """Load model performance metadata."""
    path = "models/model_metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_encoding_map():
    """Load categorical encoding mappings."""
    path = "models/encoding_map.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

model        = load_model()
metadata     = load_metadata()
encoding_map = load_encoding_map()

# ── HEADER ────────────────────────────────────────────────────
st.title("✈️ Wellness Tourism Package — Purchase Predictor")
st.markdown(
    "Predict whether a customer is likely to purchase the "
    "**Wellness Tourism Package** before contacting them."
)

# Display model performance metrics
if metadata:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", metadata.get("model_name", "N/A"))
    col2.metric("Test AUC",
                f"{metadata.get('metrics', {}).get('roc_auc', 0):.4f}")
    col3.metric("Accuracy",
                f"{metadata.get('metrics', {}).get('accuracy', 0):.4f}")
    col4.metric("F1 Score",
                f"{metadata.get('metrics', {}).get('f1_score', 0):.4f}")

st.divider()

# ── INPUT FORM ────────────────────────────────────────────────
st.subheader("📋 Enter Customer Details")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**👤 Customer Information**")
    age = st.number_input(
        "Age", min_value=18, max_value=80, value=35
    )
    gender = st.radio(
        "Gender", options=["Male", "Female"], horizontal=True
    )
    marital_status = st.selectbox(
        "Marital Status",
        options=["Single", "Married", "Divorced"]
    )
    occupation = st.selectbox(
        "Occupation",
        options=["Salaried", "Small Business",
                 "Large Business", "Free Lancer"]
    )
    designation = st.selectbox(
        "Designation",
        options=["Executive", "Manager",
                 "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.slider(
        "Monthly Income (₹)",
        min_value=5000, max_value=100000,
        value=30000, step=1000
    )
    city_tier = st.radio(
        "City Tier  (1=Metro, 3=Small city)",
        options=[1, 2, 3], horizontal=True
    )
    type_of_contact = st.radio(
        "Type of Contact",
        options=["Company Invited", "Self Inquiry"],
        horizontal=True
    )

with col_right:
    st.markdown("**🏨 Trip & Package Details**")
    num_persons = st.slider(
        "Number of Persons Visiting",
        min_value=1, max_value=10, value=2
    )
    num_children = st.slider(
        "Children Below Age 5 Visiting",
        min_value=0, max_value=5, value=0
    )
    preferred_star = st.radio(
        "Preferred Hotel Star Rating",
        options=[1, 2, 3, 4, 5], horizontal=True
    )
    num_trips = st.slider(
        "Average Trips Per Year",
        min_value=0, max_value=20, value=3
    )
    passport = st.radio(
        "Has Passport?", options=["Yes", "No"], horizontal=True
    )
    own_car = st.radio(
        "Owns a Car?", options=["Yes", "No"], horizontal=True
    )

    st.markdown("**📊 Sales Interaction Details**")
    product_pitched = st.selectbox(
        "Product Pitched",
        options=["Basic", "Standard", "Deluxe",
                 "Super Deluxe", "King"]
    )
    pitch_score = st.slider(
        "Pitch Satisfaction Score", min_value=1, max_value=5, value=3
    )
    num_followups = st.slider(
        "Number of Follow-ups", min_value=0, max_value=10, value=2
    )
    duration_pitch = st.slider(
        "Duration of Pitch (minutes)",
        min_value=5, max_value=60, value=15
    )

# ── BUILD INPUT DATAFRAME ─────────────────────────────────────

def encode_input(value, col):
    """Encode a categorical value using the saved encoding map."""
    if col in encoding_map:
        return encoding_map[col].get(str(value), 0)
    return value

def build_input_dataframe():
    """Collect all inputs and return an encoded DataFrame."""
    cat_cols = [
        "TypeofContact", "Occupation", "Gender",
        "MaritalStatus", "Designation", "ProductPitched"
    ]
    # Collect raw inputs
    raw = {
        "Age":                      age,
        "TypeofContact":            type_of_contact,
        "CityTier":                 city_tier,
        "Occupation":               occupation,
        "Gender":                   gender,
        "NumberOfPersonVisiting":   num_persons,
        "PreferredPropertyStar":    preferred_star,
        "MaritalStatus":            marital_status,
        "NumberOfTrips":            num_trips,
        "Passport":                 1 if passport == "Yes" else 0,
        "OwnCar":                   1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation":              designation,
        "MonthlyIncome":            monthly_income,
        "PitchSatisfactionScore":   pitch_score,
        "ProductPitched":           product_pitched,
        "NumberOfFollowups":        num_followups,
        "DurationOfPitch":          duration_pitch,
    }
    # Encode categorical columns
    encoded = {
        k: (encode_input(v, k) if k in cat_cols else v)
        for k, v in raw.items()
    }
    # Save inputs into a DataFrame (as required by rubric)
    return pd.DataFrame([encoded])

# ── PREDICT BUTTON ────────────────────────────────────────────
st.divider()

if st.button("🔮 Predict Purchase Likelihood", type="primary"):

    if model is None:
        st.error("Model not loaded. Please run the training pipeline.")
    else:
        # Build input dataframe
        input_df = build_input_dataframe()

        # Make prediction
        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        buy_prob   = proba[1]
        no_prob    = proba[0]

        # ── DISPLAY RESULTS ───────────────────────────────────
        st.subheader("🎯 Prediction Result")

        res_col, prob_col = st.columns(2)

        with res_col:
            if prediction == 1:
                st.success("✅  **LIKELY TO PURCHASE**")
                st.markdown(
                    f"This customer has a **{buy_prob*100:.1f}%** "
                    f"probability of purchasing the Wellness Package."
                )
            else:
                st.warning("❌  **UNLIKELY TO PURCHASE**")
                st.markdown(
                    f"This customer has only a **{buy_prob*100:.1f}%** "
                    f"probability of purchasing the Wellness Package."
                )

        with prob_col:
            st.markdown("**Probability Breakdown**")
            st.progress(float(buy_prob))
            st.markdown(
                f"Purchase:    **{buy_prob*100:.1f}%** &nbsp;|&nbsp; "
                f"No Purchase: **{no_prob*100:.1f}%**"
            )

        # ── BUSINESS RECOMMENDATION ───────────────────────────
        st.subheader("💼 Business Recommendation")
        if prediction == 1:
            st.info(
                "**Priority Action:** Schedule a personalised sales call.\n\n"
                "Offer a tailored Wellness Tourism Package matching "
                "the customer's profile and preferences.\n\n"
                "Highlight premium features relevant to their "
                "income level and travel history."
            )
        else:
            st.info(
                "**Lower Priority:** Send informational content "
                "about wellness tourism benefits.\n\n"
                "Follow up in 30 days with a promotional or "
                "discounted offer to increase interest.\n\n"
                "Consider nurturing through email campaigns "
                "before making direct contact."
            )

        # ── CUSTOMER PROFILE SUMMARY ──────────────────────────
        st.subheader("📊 Customer Profile Summary")
        summary_df = pd.DataFrame({
            "Feature": [
                "Age", "Gender", "Marital Status", "Occupation",
                "Monthly Income", "City Tier", "Passport",
                "Annual Trips", "Pitch Satisfaction",
                "Follow-ups", "Product Pitched"
            ],
            "Value": [
                age, gender, marital_status, occupation,
                f"₹{monthly_income:,}", city_tier,
                passport, num_trips, pitch_score,
                num_followups, product_pitched
            ]
        })
        st.dataframe(summary_df, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "Wellness Tourism Package Predictor &nbsp;|&nbsp; "
    "MLOps Pipeline &nbsp;|&nbsp; "
    "GitHub Actions CI/CD &nbsp;|&nbsp; "
    "Streamlit Community Cloud"
    "</div>",
    unsafe_allow_html=True
)
