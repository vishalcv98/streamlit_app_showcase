import streamlit as st
import pandas as pd
import pickle




def load_model():
    with open("xgb_car_price_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🚗 Car Price Prediction")

st.write("Enter car details below:")

# -------------------------------
# User Inputs
# -------------------------------
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=50000)
mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0, value=18.0)
age = st.number_input("Car Age (years)", min_value=0, max_value=30, value=5)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])

# -------------------------------
# One-hot encode fuel (MATCH TRAINING)
# -------------------------------
fuel_encoding = {
    "Petrol":   [1, 0, 0],
    "Diesel":   [0, 1, 0],
    "Electric": [0, 0, 1]
}

petrol, diesel, electric = fuel_encoding[fuel_type]

# -------------------------------
# Create input DataFrame
# -------------------------------
input_df = pd.DataFrame(
    [[km_driven, mileage, age, petrol, diesel, electric]],
    columns=['km_driven','mileage','age','Petrol','Diesel','Electric']
)

st.write("### Model Input")
st.dataframe(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Selling Price: ₹ {round(prediction, 2)}")