import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Memuat model dan scaler
with open('best_xgb_model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Fitur input (pastikan sudah di-encode sesuai pelatihan)
input_features = [35, 1, 1, 50000, 5, 0, 15000, 0, 5.0, 30.0, 10, 700, 0]

categorical_features = input_features[1:4]
numerical_features = input_features[0:1] + input_features[3:5] + input_features[6:8] + input_features[9:]

features = categorical_features + numerical_features
features_df = pd.DataFrame([features])
scaled_features = scaler.transform(features_df)

prediction = model.predict(scaled_features)
status = "Loan Approved" if prediction[0] == 1 else "Loan Denied"

print("Predicted Loan Status:", status)
