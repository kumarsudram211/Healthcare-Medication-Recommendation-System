import streamlit as st
import pandas as pd
import ast
import pickle

from sklearn.preprocessing import LabelEncoder

# Reconstructing the disease list in the correct order
diseases = [
    '(vertigo) Paroymsal  Positional Vertigo',
    'AIDS',
    'Acne',
    'Alcoholic hepatitis',
    'Allergy',
    'Arthritis',
    'Bronchial Asthma',
    'Cervical spondylosis',
    'Chicken pox',
    'Chronic cholestasis',
    'Common Cold',
    'Dengue',
    'Diabetes ',
    'Dimorphic hemmorhoids(piles)',
    'Drug Reaction',
    'Fungal infection',
    'GERD',
    'Gastroenteritis',
    'Heart attack',
    'Hepatitis B',
    'Hepatitis C',
    'Hepatitis D',
    'Hepatitis E',
    'Hypertension ',
    'Hyperthyroidism',
    'Hypoglycemia',
    'Hypothyroidism',
    'Impetigo',
    'Jaundice',
    'Malaria',
    'Migraine',
    'Osteoarthristis',
    'Paralysis (brain hemorrhage)',
    'Peptic ulcer diseae',
    'Pneumonia',
    'Psoriasis',
    'Tuberculosis',
    'Typhoid',
    'Urinary tract infection',
    'Varicose veins',
    'hepatitis A'
]

# Creating and fitting the LabelEncoder
le = LabelEncoder()
le.fit(diseases)

# Loading model
with open(r"C:\Users\SANDILYA SUNDRAM\Desktop\Revision\archive\Medicine app\SVC.pkl", 'rb') as file:
    model = pickle.load(file)

# Loading mapping data
description_df = pd.read_csv(r"C:\Users\SANDILYA SUNDRAM\Desktop\Revision\archive\Medicine app\description.csv")
diet_df = pd.read_csv(r"C:\Users\SANDILYA SUNDRAM\Desktop\Revision\archive\Medicine app\diets.csv")
medications_df = pd.read_csv(r"C:\Users\SANDILYA SUNDRAM\Desktop\Revision\archive\Medicine app\medications.csv")
precautions_df = pd.read_csv(r"C:\Users\SANDILYA SUNDRAM\Desktop\Revision\archive\Medicine app\precautions.csv")
workout_df = pd.read_csv(r"C:\Users\SANDILYA SUNDRAM\Desktop\Revision\archive\Medicine app\workout.csv")

# Helper functions to fetch data
def get_description(disease):
    row = description_df[description_df['Disease'] == disease]
    return row['Description'].values[0] if not row.empty else "No description found."

def get_list_column(df, disease, column):
    row = df[df['Disease'] == disease]
    if not row.empty:
        try:
            return ast.literal_eval(row[column].values[0])
        except:
            return ["Parsing error"]
    return ["Not available"]

def get_precautions(disease):
    row = precautions_df[precautions_df['Disease'] == disease]
    if not row.empty:
        return row.iloc[0, 1:].tolist()
    return ["No precautions found."]

def get_workout(disease):
    row = workout_df[workout_df['Disease'] == disease]
    return row['Workout'].values[0] if not row.empty else "No workout found."

# Streamlit UI
st.title("🩺 Disease Prediction & Recommendations")

user_input = st.text_input("Enter symptoms (comma-separated)", placeholder="e.g., fever, cough, headache")

# This goes near the top, below your imports and model loading
all_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
    'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
    'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
    'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
    'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
    'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
    'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
    'red_sore_around_nose', 'yellow_crust_ooze'
]
def symptoms_to_vector(symptoms_input):
    user_symptoms = [sym.strip().lower() for sym in symptoms_input.split(',')]
    return [1 if symptom in user_symptoms else 0 for symptom in all_symptoms]


if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some symptoms.")
    else:
        try:
            input_vector = symptoms_to_vector(user_input)
            predicted_index = model.predict([input_vector])[0]
            predicted_disease = le.inverse_transform([predicted_index])[0]  # Convert number → disease
            st.success(f"✅ Predicted Disease: **{predicted_disease}**")

            st.subheader("📝 Description")
            st.write(get_description(predicted_disease))

            st.subheader("💊 Medications")
            for med in get_list_column(medications_df, predicted_disease, 'Medications'):
                st.write(f"- {med}")

            st.subheader("🥗 Diet Recommendations")
            for diet in get_list_column(diet_df, predicted_disease, 'Diet'):
                st.write(f"- {diet}")

            st.subheader("⚠️ Precautions")
            for p in get_precautions(predicted_disease):
                st.write(f"- {p}")

            st.subheader("🏃 Workout Plan")
            st.write(get_workout(predicted_disease))

        except Exception as e:
            st.error(f"Prediction failed: {e}")