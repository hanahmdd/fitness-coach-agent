import streamlit as st
import pandas as pd
import os
import sys
import traceback

# Use pathlib for more robust path handling
from pathlib import Path

# Modify the import to use local utils
from utils import load_model, predict_with_model

# Use a relative path for the model directory
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)

# Debugging: Check if model files exist
def check_model_files():
    model_files = ["diet_model.pkl", "gym_model.pkl"]
    results = {}

    for file in model_files:
        path = MODEL_DIR / file
        results[file] = path.exists()

    return results


# Function to calculate BMI
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m * height_m), 2)


# Streamlit Page Configuration
st.set_page_config(page_title="Fitness Coach Agent", page_icon="🏋️", layout="centered")

st.title("🏋️‍♂️ Fitness Coach Agent")
st.markdown("Welcome! Get your personalized **workout** and **diet plan** based on your profile 💪")

# Debug section
with st.expander("Debug Information"):
    st.write("Model Files Check:")
    model_check = check_model_files()
    for file, exists in model_check.items():
        status = "✅ Found" if exists else "❌ Missing"
        st.write(f"{file}: {status}")

    st.write("Current Directory:", os.getcwd())
    st.write("Python Version:", sys.version)

# Try to load the models
models_loaded = {"diet": False, "gym": False}
try:
    diet_model = load_model("diet_model.pkl")
    models_loaded["diet"] = True
    st.success("Diet model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load diet model: {str(e)}")

try:
    gym_model = load_model("gym_model.pkl")
    models_loaded["gym"] = True
    st.success("Gym model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load gym model: {str(e)}")

# Form to collect user input
with st.form("user_form"):
    st.subheader("👤 Basic Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 10, 100, 25)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
        fitness_goal = st.selectbox("Fitness Goal", ["Lose Weight", "Gain Muscle", "Maintain Fitness"])

    # Calculate BMI
    bmi = calculate_bmi(weight, height)
    st.write(f"Calculated BMI: {bmi}")

    # Health Information
    st.subheader("🩺 Health Information")

    col1, col2 = st.columns(2)
    with col1:
        disease_type = st.multiselect("Medical Conditions",
                                      ["None", "Diabetes", "Hypertension", "Obesity"],
                                      default=["None"])

        severity = st.selectbox("Condition Severity",
                                ["None", "Mild", "Moderate", "Severe"],
                                index=0)

    with col2:
        activity_level = st.selectbox("Physical Activity Level",
                                      ["Sedentary", "Moderate", "Active"],
                                      index=1)

    # Dietary Preferences
    st.subheader("🍽️ Dietary Preferences")

    col1, col2 = st.columns(2)
    with col1:
        dietary_restrictions = st.multiselect("Dietary Restrictions",
                                              ["None", "Low_Sodium", "Low_Sugar"],
                                              default=["None"])

        allergies = st.multiselect("Allergies",
                                   ["None", "Gluten", "Peanuts"],
                                   default=["None"])

    with col2:
        preferred_cuisine = st.multiselect("Preferred Cuisine",
                                           ["None", "Chinese", "Indian", "Italian", "Mexican"],
                                           default=["None"])

    submitted = st.form_submit_button("💡 Get Recommendations")

# When the form is submitted
if submitted:
    if not (models_loaded["diet"] or models_loaded["gym"]):
        st.warning("Cannot generate recommendations because models failed to load.")
    else:
        # Process inputs to handle multiselects properly
        has_diabetes = 1 if "Diabetes" in disease_type and "None" not in disease_type else 0
        has_hypertension = 1 if "Hypertension" in disease_type and "None" not in disease_type else 0
        has_obesity = 1 if "Obesity" in disease_type and "None" not in disease_type else 0

        is_mild = 1 if severity == "Mild" else 0
        is_moderate = 1 if severity == "Moderate" else 0
        is_severe = 1 if severity == "Severe" else 0

        is_active = 1 if activity_level == "Active" else 0
        is_moderate_activity = 1 if activity_level == "Moderate" else 0
        is_sedentary = 1 if activity_level == "Sedentary" else 0

        has_low_sodium = 1 if "Low_Sodium" in dietary_restrictions and "None" not in dietary_restrictions else 0
        has_low_sugar = 1 if "Low_Sugar" in dietary_restrictions and "None" not in dietary_restrictions else 0

        has_gluten_allergy = 1 if "Gluten" in allergies and "None" not in allergies else 0
        has_peanut_allergy = 1 if "Peanuts" in allergies and "None" not in allergies else 0

        likes_chinese = 1 if "Chinese" in preferred_cuisine and "None" not in preferred_cuisine else 0
        likes_indian = 1 if "Indian" in preferred_cuisine and "None" not in preferred_cuisine else 0
        likes_italian = 1 if "Italian" in preferred_cuisine and "None" not in preferred_cuisine else 0
        likes_mexican = 1 if "Mexican" in preferred_cuisine and "None" not in preferred_cuisine else 0

        # Create user input with all required fields for the model
        user_input = {
            # Basic info
            "Age": age,
            "Gender_Male": 1 if gender == "Male" else 0,
            "Gender_Female": 1 if gender == "Female" else 0,
            "Weight_kg": weight,
            "Height_cm": height,
            "BMI": bmi,

            # Also include alternate column names that might be in the model
            "Weight": weight,
            "Height": height,
            "Sex_Male": 1 if gender == "Male" else 0,
            "Sex_Female": 1 if gender == "Female" else 0,

            # Health conditions
            "Disease_Type_Diabetes": has_diabetes,
            "Disease_Type_Hypertension": has_hypertension,
            "Disease_Type_Obesity": has_obesity,
            "Diabetes_Yes": has_diabetes,
            "Hypertension_Yes": has_hypertension,

            # Condition severity
            "Severity_Mild": is_mild,
            "Severity_Moderate": is_moderate,
            "Severity_Severe": is_severe,

            # Activity level
            "Physical_Activity_Level_Active": is_active,
            "Physical_Activity_Level_Moderate": is_moderate_activity,
            "Physical_Activity_Level_Sedentary": is_sedentary,

            # Dietary preferences
            "Dietary_Restrictions_Low_Sodium": has_low_sodium,
            "Dietary_Restrictions_Low_Sugar": has_low_sugar,
            "Allergies_Gluten": has_gluten_allergy,
            "Allergies_Peanuts": has_peanut_allergy,

            # Cuisine preferences
            "Preferred_Cuisine_Chinese": likes_chinese,
            "Preferred_Cuisine_Indian": likes_indian,
            "Preferred_Cuisine_Italian": likes_italian,
            "Preferred_Cuisine_Mexican": likes_mexican,

            # Fitness goals with multiple naming variations
            "Fitness Goal_Lose Weight": 1 if fitness_goal == "Lose Weight" else 0,
            "Fitness Goal_Gain Muscle": 1 if fitness_goal == "Gain Muscle" else 0,
            "Fitness Goal_Maintain Fitness": 1 if fitness_goal == "Maintain Fitness" else 0,
            "Fitness Goal_Weight Loss": 1 if fitness_goal == "Lose Weight" else 0,
            "Fitness Goal_Weight Gain": 1 if fitness_goal == "Gain Muscle" else 0,
        }

        # Display input data for debugging
        with st.expander("Debug: Input Data"):
            st.write(user_input)

            # Also show a nicely formatted summary of selections
            st.subheader("Your Selections Summary")
            st.write(f"- Age: {age} years")
            st.write(f"- Gender: {gender}")
            st.write(f"- Height: {height} cm")
            st.write(f"- Weight: {weight} kg")
            st.write(f"- BMI: {bmi}")
            st.write(f"- Fitness Goal: {fitness_goal}")

            if "None" not in disease_type:
                st.write(f"- Medical Conditions: {', '.join([d for d in disease_type if d != 'None'])}")
            if severity != "None":
                st.write(f"- Severity: {severity}")
            st.write(f"- Activity Level: {activity_level}")

            if "None" not in dietary_restrictions:
                st.write(f"- Dietary Restrictions: {', '.join([r for r in dietary_restrictions if r != 'None'])}")
            if "None" not in allergies:
                st.write(f"- Allergies: {', '.join([a for a in allergies if a != 'None'])}")
            if "None" not in preferred_cuisine:
                st.write(f"- Preferred Cuisines: {', '.join([c for c in preferred_cuisine if c != 'None'])}")

        # Try to predict plans
        st.success("✅ Here are your personalized plans!")

        # Predict the gym plan
        if models_loaded["gym"]:
            st.subheader("🏃 Your Workout Recommendation")
            try:
                gym_plan = predict_with_model(gym_model, user_input)
                st.markdown(f"**Plan:** {gym_plan}")
            except Exception as e:
                st.error(f"Couldn't generate workout plan: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Workout recommendation not available (model not loaded)")

        # Predict the diet plan
        if models_loaded["diet"]:
            st.subheader("🥗 Your Diet Recommendation")
            try:
                diet_plan = predict_with_model(diet_model, user_input)
                st.markdown(f"**Plan:** {diet_plan}")
            except Exception as e:
                st.error(f"Couldn't generate diet plan: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Diet recommendation not available (model not loaded)")

        # Add additional recommendations based on the user's inputs
        st.subheader("📝 Additional Recommendations")

        # BMI-based recommendations
        if bmi < 18.5:
            st.write("- Based on your BMI, consider focusing on nutrient-dense foods to reach a healthy weight.")
        elif bmi >= 25 and bmi < 30:
            st.write("- Based on your BMI, consider moderate calorie restriction and increased physical activity.")
        elif bmi >= 30:
            st.write(
                "- Based on your BMI, consider consulting with a healthcare provider for a personalized weight management plan.")

        # Disease-specific recommendations
        if has_diabetes:
            st.write("- For diabetes management, monitor carbohydrate intake and focus on low glycemic index foods.")
        if has_hypertension:
            st.write("- For hypertension management, consider the DASH diet with reduced sodium intake.")

        # Activity level recommendations
        if is_sedentary:
            st.write("- Consider gradually increasing your daily activity with short walks or light exercises.")

else:
    st.info("Fill in the info above and click 'Get Recommendations'")

st.markdown("---")
st.caption("Developed by Hanaa • Fitness Coach Agent")
