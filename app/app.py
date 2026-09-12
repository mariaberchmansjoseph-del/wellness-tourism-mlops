import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── LOAD MODEL AND METADATA ───────────────────────────────────

def load_model():
    path = "models/best_model.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_metadata():
    path = "models/model_metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_encoding_map():
    path = "models/encoding_map.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

model        = load_model()
metadata     = load_metadata()
encoding_map = load_encoding_map()

# ── ENCODE CATEGORICAL INPUT ──────────────────────────────────

def encode_value(value, col):
    if col in encoding_map:
        return encoding_map[col].get(str(value), 0)
    return value

# ── PREDICTION FUNCTION ───────────────────────────────────────

def predict_purchase(
    age,
    gender,
    marital_status,
    occupation,
    designation,
    monthly_income,
    city_tier,
    type_of_contact,
    num_persons,
    num_children,
    preferred_star,
    num_trips,
    passport,
    own_car,
    product_pitched,
    pitch_score,
    num_followups,
    duration_pitch,
):
    if model is None:
        return (
            "❌ Model not loaded. Please run the training pipeline.",
            "N/A",
            "N/A",
            "N/A",
        )

    cat_cols = [
        "TypeofContact", "Occupation", "Gender",
        "MaritalStatus", "Designation", "ProductPitched"
    ]

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

    encoded = {
        k: (encode_value(v, k) if k in cat_cols else v)
        for k, v in raw.items()
    }

    input_df   = pd.DataFrame([encoded])
    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    buy_prob   = round(float(proba[1]) * 100, 1)
    no_prob    = round(float(proba[0]) * 100, 1)

    # ── RESULT TEXT ───────────────────────────────────────────
    if prediction == 1:
        result = (
            f"✅  LIKELY TO PURCHASE\n\n"
            f"This customer has a {buy_prob}% probability of purchasing "
            f"the Wellness Tourism Package."
        )
        recommendation = (
            "🎯  Priority Action: Schedule a personalised sales call.\n"
            "     Offer a tailored Wellness Package matching their profile.\n"
            "     Highlight premium features relevant to their income and preference."
        )
    else:
        result = (
            f"❌  UNLIKELY TO PURCHASE\n\n"
            f"This customer has only a {buy_prob}% probability of purchasing "
            f"the Wellness Tourism Package."
        )
        recommendation = (
            "📋  Lower Priority: Send informational content about wellness tourism.\n"
            "     Follow up in 30 days with a discounted or promotional offer.\n"
            "     Consider nurturing through email before direct contact."
        )

    probability_summary = (
        f"Purchase Probability:     {buy_prob}%\n"
        f"No Purchase Probability:  {no_prob}%"
    )

    model_info = (
        f"Model Used:   {metadata.get('model_name', 'N/A')}\n"
        f"Test AUC:     {metadata.get('metrics', {}).get('roc_auc', 'N/A')}\n"
        f"Accuracy:     {metadata.get('metrics', {}).get('accuracy', 'N/A')}\n"
        f"F1 Score:     {metadata.get('metrics', {}).get('f1_score', 'N/A')}"
    )

    return result, probability_summary, recommendation, model_info


# ── BUILD GRADIO INTERFACE ────────────────────────────────────

with gr.Blocks(
    title="Wellness Tourism Package Predictor",
    theme=gr.themes.Soft(),
) as demo:

    # Header
    gr.Markdown("""
    # ✈️ Wellness Tourism Package — Purchase Predictor
    **MLOps Pipeline | Visit with Us | Powered by GitHub Actions + HuggingFace**

    Predict whether a customer is likely to purchase the Wellness Tourism Package
    before contacting them. Fill in the customer details below and click **Predict**.
    """)

    gr.Markdown("---")

    # ── INPUT SECTION ─────────────────────────────────────────
    with gr.Row():

        # Left column — Customer Information
        with gr.Column():
            gr.Markdown("### 👤 Customer Information")

            age = gr.Slider(
                minimum=18, maximum=80, value=35, step=1,
                label="Age"
            )
            gender = gr.Radio(
                choices=["Male", "Female"],
                value="Male",
                label="Gender"
            )
            marital_status = gr.Dropdown(
                choices=["Single", "Married", "Divorced"],
                value="Single",
                label="Marital Status"
            )
            occupation = gr.Dropdown(
                choices=["Salaried", "Small Business",
                         "Large Business", "Free Lancer"],
                value="Salaried",
                label="Occupation"
            )
            designation = gr.Dropdown(
                choices=["Executive", "Manager",
                         "Senior Manager", "AVP", "VP"],
                value="Executive",
                label="Designation"
            )
            monthly_income = gr.Slider(
                minimum=5000, maximum=100000,
                value=30000, step=1000,
                label="Monthly Income (₹)"
            )
            city_tier = gr.Radio(
                choices=[1, 2, 3],
                value=1,
                label="City Tier  (1 = Metro, 3 = Small city)"
            )
            type_of_contact = gr.Radio(
                choices=["Company Invited", "Self Inquiry"],
                value="Company Invited",
                label="Type of Contact"
            )

        # Right column — Trip and Sales Details
        with gr.Column():
            gr.Markdown("### 🏨 Trip & Package Details")

            num_persons = gr.Slider(
                minimum=1, maximum=10, value=2, step=1,
                label="Number of Persons Visiting"
            )
            num_children = gr.Slider(
                minimum=0, maximum=5, value=0, step=1,
                label="Children Below Age 5 Visiting"
            )
            preferred_star = gr.Radio(
                choices=[1, 2, 3, 4, 5],
                value=3,
                label="Preferred Hotel Star Rating"
            )
            num_trips = gr.Slider(
                minimum=0, maximum=20, value=3, step=1,
                label="Average Trips Per Year"
            )
            passport = gr.Radio(
                choices=["Yes", "No"],
                value="No",
                label="Has Passport?"
            )
            own_car = gr.Radio(
                choices=["Yes", "No"],
                value="No",
                label="Owns a Car?"
            )

            gr.Markdown("### 📊 Sales Interaction Details")

            product_pitched = gr.Dropdown(
                choices=["Basic", "Standard", "Deluxe",
                         "Super Deluxe", "King"],
                value="Basic",
                label="Product Pitched"
            )
            pitch_score = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Pitch Satisfaction Score"
            )
            num_followups = gr.Slider(
                minimum=0, maximum=10, value=2, step=1,
                label="Number of Follow-ups"
            )
            duration_pitch = gr.Slider(
                minimum=5, maximum=60, value=15, step=1,
                label="Duration of Pitch (minutes)"
            )

    gr.Markdown("---")

    # ── PREDICT BUTTON ────────────────────────────────────────
    predict_btn = gr.Button(
        "🔮  Predict Purchase Likelihood",
        variant="primary",
        size="lg"
    )

    gr.Markdown("---")

    # ── OUTPUT SECTION ────────────────────────────────────────
    gr.Markdown("### 🎯 Prediction Results")

    with gr.Row():
        with gr.Column():
            result_output = gr.Textbox(
                label="Prediction",
                lines=4,
                interactive=False
            )
            prob_output = gr.Textbox(
                label="Probability Breakdown",
                lines=3,
                interactive=False
            )
        with gr.Column():
            rec_output = gr.Textbox(
                label="Business Recommendation",
                lines=5,
                interactive=False
            )
            model_output = gr.Textbox(
                label="Model Information",
                lines=5,
                interactive=False
            )

    # ── WIRE UP BUTTON ────────────────────────────────────────
    predict_btn.click(
        fn=predict_purchase,
        inputs=[
            age, gender, marital_status, occupation,
            designation, monthly_income, city_tier,
            type_of_contact, num_persons, num_children,
            preferred_star, num_trips, passport, own_car,
            product_pitched, pitch_score, num_followups,
            duration_pitch,
        ],
        outputs=[
            result_output,
            prob_output,
            rec_output,
            model_output,
        ],
    )

    gr.Markdown("---")

    # ── EXAMPLE INPUTS ────────────────────────────────────────
    gr.Markdown("### 📋 Example Customers (click to auto-fill)")
    gr.Examples(
        examples=[
            # age, gender, marital, occupation, designation,
            # income, city_tier, contact, persons, children,
            # star, trips, passport, car, product,
            # pitch_score, followups, duration
            [
                45, "Male", "Married", "Salaried", "Senior Manager",
                60000, 1, "Self Inquiry", 4, 1,
                4, 5, "Yes", "Yes", "Deluxe",
                4, 3, 20
            ],
            [
                28, "Female", "Single", "Free Lancer", "Executive",
                18000, 3, "Company Invited", 2, 0,
                2, 1, "No", "No", "Basic",
                2, 1, 8
            ],
        ],
        inputs=[
            age, gender, marital_status, occupation,
            designation, monthly_income, city_tier,
            type_of_contact, num_persons, num_children,
            preferred_star, num_trips, passport, own_car,
            product_pitched, pitch_score, num_followups,
            duration_pitch,
        ],
        label="Click an example to auto-fill all inputs"
    )

    gr.Markdown("""
    ---
    *Wellness Tourism Predictor · MLOps Pipeline · GitHub Actions CI/CD*
    """)


# ── LAUNCH ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
