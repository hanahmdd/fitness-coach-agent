import joblib
import os
import pandas as pd
import pickle
import numpy as np

# Base directory = where utils.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_model(filename):
    """Load a model from the model directory."""
    model_path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load and return the model
    try:
        with open(model_path, 'rb') as f:
            # Try both joblib and pickle
            try:
                model = joblib.load(f)
            except:
                # Rewind file pointer
                f.seek(0)
                model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model '{filename}': {str(e)}")


def predict_with_model(model, input_data):
    """Make a prediction using the trained model."""
    try:
        model_instance, label_encoder, feature_columns = model

        # Debug information
        print(f"Input data keys: {list(input_data.keys())}")
        print(f"Expected feature columns: {feature_columns}")

        # Determine which model we're dealing with based on the feature columns
        is_diet_model = 'Weight_kg' in feature_columns or any('Dietary_Restrictions' in col for col in feature_columns)

        # Create a DataFrame with zeros for all model features
        final_input = pd.DataFrame(0, index=[0], columns=feature_columns)

        # For Diet Model
        if is_diet_model:
            print("Using Diet Model mapping")

            # Map the direct matching columns
            for col in feature_columns:
                if col in input_data:
                    final_input[col] = input_data[col]
                    continue

                # Handle gender columns
                if col.startswith('Gender_'):
                    gender_val = col.split('_')[1]  # Get Male or Female
                    if f'Gender_{gender_val}' in input_data and input_data[f'Gender_{gender_val}'] == 1:
                        final_input[col] = 1
                    continue

                # Handle disease columns
                if col.startswith('Disease_Type_'):
                    disease = col.split('_')[-1]  # Get disease name
                    if f'Disease_Type_{disease}' in input_data and input_data[f'Disease_Type_{disease}'] == 1:
                        final_input[col] = 1
                    continue

                # Handle severity columns
                if col.startswith('Severity_'):
                    severity_val = col.split('_')[1]  # Get severity level
                    if f'Severity_{severity_val}' in input_data and input_data[f'Severity_{severity_val}'] == 1:
                        final_input[col] = 1
                    continue

                # Handle activity level columns
                if col.startswith('Physical_Activity_Level_'):
                    activity = col.split('_')[-1]  # Get activity level
                    if f'Physical_Activity_Level_{activity}' in input_data and input_data[
                        f'Physical_Activity_Level_{activity}'] == 1:
                        final_input[col] = 1
                    continue

                # Handle dietary restrictions
                if col.startswith('Dietary_Restrictions_'):
                    restriction = col.split('_')[-1]  # Get restriction
                    if f'Dietary_Restrictions_{restriction}' in input_data and input_data[
                        f'Dietary_Restrictions_{restriction}'] == 1:
                        final_input[col] = 1
                    continue

                # Handle allergies
                if col.startswith('Allergies_'):
                    allergy = col.split('_')[1]  # Get allergy
                    if f'Allergies_{allergy}' in input_data and input_data[f'Allergies_{allergy}'] == 1:
                        final_input[col] = 1
                    continue

                # Handle preferred cuisine
                if col.startswith('Preferred_Cuisine_'):
                    cuisine = col.split('_')[2]  # Get cuisine
                    if f'Preferred_Cuisine_{cuisine}' in input_data and input_data[f'Preferred_Cuisine_{cuisine}'] == 1:
                        final_input[col] = 1
                    continue

        # For Gym Model
        else:
            print("Using Gym Model mapping")

            # Handle basic metrics
            for basic_col in ['Age', 'BMI']:
                if basic_col in feature_columns and basic_col in input_data:
                    final_input[basic_col] = input_data[basic_col]

            # Handle height and weight
            if 'Height' in feature_columns:
                if 'Height' in input_data:
                    final_input['Height'] = input_data['Height']
                elif 'Height_cm' in input_data:
                    final_input['Height'] = input_data['Height_cm']

            if 'Weight' in feature_columns:
                if 'Weight' in input_data:
                    final_input['Weight'] = input_data['Weight']
                elif 'Weight_kg' in input_data:
                    final_input['Weight'] = input_data['Weight_kg']

            # Handle gender/sex columns
            for col in feature_columns:
                if col.startswith('Sex_'):
                    gender_val = col.split('_')[1]  # Get Male or Female

                    # Try to map from either Sex_ or Gender_ fields in input
                    if f'Sex_{gender_val}' in input_data and input_data[f'Sex_{gender_val}'] == 1:
                        final_input[col] = 1
                    elif f'Gender_{gender_val}' in input_data and input_data[f'Gender_{gender_val}'] == 1:
                        final_input[col] = 1

            # Handle hypertension and diabetes
            for col in feature_columns:
                if col == 'Hypertension_Yes':
                    if 'Disease_Type_Hypertension' in input_data:
                        final_input[col] = input_data['Disease_Type_Hypertension']
                    elif 'Hypertension_Yes' in input_data:
                        final_input[col] = input_data['Hypertension_Yes']

                if col == 'Hypertension_No':
                    if 'Disease_Type_Hypertension' in input_data:
                        final_input[col] = 1 - input_data['Disease_Type_Hypertension']
                    elif 'Hypertension_No' in input_data:
                        final_input[col] = input_data['Hypertension_No']

                if col == 'Diabetes_Yes':
                    if 'Disease_Type_Diabetes' in input_data:
                        final_input[col] = input_data['Disease_Type_Diabetes']
                    elif 'Diabetes_Yes' in input_data:
                        final_input[col] = input_data['Diabetes_Yes']

                if col == 'Diabetes_No':
                    if 'Disease_Type_Diabetes' in input_data:
                        final_input[col] = 1 - input_data['Disease_Type_Diabetes']
                    elif 'Diabetes_No' in input_data:
                        final_input[col] = input_data['Diabetes_No']

            # Handle fitness goals
            for col in feature_columns:
                if col == 'Fitness Goal_Weight Loss' and 'Fitness Goal_Lose Weight' in input_data:
                    final_input[col] = input_data['Fitness Goal_Lose Weight']
                elif col == 'Fitness Goal_Weight Gain' and 'Fitness Goal_Gain Muscle' in input_data:
                    final_input[col] = input_data['Fitness Goal_Gain Muscle']
                elif col in input_data:
                    final_input[col] = input_data[col]

        # Debugging output
        print(f"Final input shape: {final_input.shape}")
        print(f"Final input columns: {list(final_input.columns)}")

        # Check for missing values and fill with 0
        if final_input.isnull().values.any():
            print("Warning: Input contains missing values. Filling with zeros.")
            final_input.fillna(0, inplace=True)

        # Make prediction
        prediction = model_instance.predict(final_input)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return predicted_label

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise Exception(f"Prediction failed: {str(e)}")