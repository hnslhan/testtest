import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

with open('best_xgb_model.pkl', 'rb') as model_file:
    loaded_xgb_model = pickle.load(model_file)
    
print(predictions)
