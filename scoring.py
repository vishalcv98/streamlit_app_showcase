import pickle
import pandas as pd

# Load model once
with open("xgb_car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

def score_model(km_driven, mileage, age, petrol, diesel,electric):

    X = pd.DataFrame([[
        km_driven,
        mileage,
        age,
        petrol,
        diesel,
        electric]], columns=[
        'km_driven',
        'mileage',
        'age',
        'Petrol',
        'Diesel',
        'Electric'
    ])

    # Predict
    prediction = model.predict(X)

    return float(prediction[0])