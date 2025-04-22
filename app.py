import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.title('Loan Prediction App')

# Memuat model yang sudah dilatih
model = XGBClassifier()
model.load_model('best_xgb_model.json')

# Memuat scaler yang sudah dilatih
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict_loan_status(features):
    try:
        # features[0:4] are categorical features already encoded
        categorical_encoded = features[:4]
        numerical_features = features[4:]

        # Scale numerical features
        numerical_df = pd.DataFrame([numerical_features])
        scaled_numerical = scaler.transform(numerical_df)

        # Combine all features
        final_features = categorical_encoded + scaled_numerical[0].tolist()
        features_df = pd.DataFrame([final_features])

        # Predict
        prediction = model.predict(features_df)
        return prediction[0]

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


# Input dari pengguna
person_age = st.number_input('Age of the Person', min_value=18, max_value=100, step=1)
person_gender = st.selectbox('Gender of the Person', ['Male', 'Female'])
person_education = st.selectbox('Education Level', ['High School', 'Bachelors', 'Masters', 'PhD'])
person_income = st.number_input('Annual Income', min_value=0, step=1000)
person_emp_exp = st.number_input('Years of Work Experience', min_value=0, step=1)
person_home_ownership = st.selectbox('Home Ownership Status', ['Own', 'Rent', 'Mortgage'])
loan_amnt = st.number_input('Loan Amount', min_value=0, step=500)
loan_intent = st.selectbox('Loan Intent', ['Personal', 'Business', 'Debt Consolidation'])
loan_int_rate = st.number_input('Loan Interest Rate (%)', min_value=0.0, step=0.1)
loan_percent_income = st.number_input('Loan as Percentage of Income', min_value=0.0, max_value=100.0, step=0.1)
cb_person_cred_hist_length = st.number_input('Credit History Length (in years)', min_value=0, step=1)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, step=1)
previous_loan_defaults_on_file = st.selectbox('Previous Loan Defaults', [0, 1])

input_features = [
    person_age,
    person_gender,
    person_education,
    person_income,
    person_emp_exp,
    person_home_ownership,
    loan_amnt,
    loan_intent,
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    credit_score,
    previous_loan_defaults_on_file
]

# Kategori fitur yang perlu dienkode
categorical_features = {
    'person_gender': {'Male': 1, 'Female': 0},
    'person_education': {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3},
    'person_home_ownership': {'Own': 0, 'Rent': 1, 'Mortgage': 2},
    'loan_intent': {'Personal': 0, 'Business': 1, 'Debt Consolidation': 2}
}

# Encode fitur input
# Features are already encoded before being passed in
categorical_encoded = features[:4]
numerical_features = features[4:]

numerical_df = pd.DataFrame([numerical_features])
scaled_numerical = scaler.transform(numerical_df)

final_features = categorical_encoded + scaled_numerical[0].tolist()
features_df = pd.DataFrame([final_features])


# Prediksi jika tombol ditekan
if st.button('Predict Loan Status'):
    st.write("Predicting loan status...")
    prediction = predict_loan_status(encoded_features)  # Pastikan mengirimkan `encoded_features`
    
    if prediction is not None:
        if prediction == 1:
            st.success('Loan Approved')
        else:
            st.error('Loan Denied')
