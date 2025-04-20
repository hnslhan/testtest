import pickle
import numpy as np
import pandas as pd

with open('best_model.pkl', 'rb') as file:
    saved_objects = pickle.load(file)
    model = saved_objects['model']
    encoder = saved_objects['encoder']

columns = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
           'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
           'lead_time', 'arrival_year', 'arrival_month', 'arrival_date',
           'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
           'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']

def predict_booking_status(input_data):
    df = pd.DataFrame([input_data], columns=columns)

    # Identify categorical columns (object types)
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = encoder.transform(df[cat_cols])

    prediction = model.predict(df)
    return prediction[0]
