import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

class LoanPredictionModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.model = XGBClassifier(random_state=42, n_estimators=50, eval_metric='logloss')
        self.scaler = StandardScaler()

    def preprocess_data(self):
        # Drop the target column 'loan_status' and define features (X) and target (y)
        X = self.data.drop(columns=['loan_status'])
        y = self.data['loan_status']
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit and transform the training data, and transform the test data
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self):
        # Train the model with the preprocessed data
        self.model.fit(self.X_train_scaled, self.y_train)

    def evaluate_model(self):
        # Predict and evaluate the model performance
        predictions = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"XGBoost Accuracy: {accuracy}")

        # Save both the model and the scaler for future use
        with open('best_xgb_model.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)
        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(self.scaler, scaler_file)

# Example usage
data_path = 'Dataset_A_loan.csv'  # Update the path to your dataset if needed
model = LoanPredictionModel(data_path)
model.preprocess_data()
model.train_model()
model.evaluate_model()