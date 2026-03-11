import streamlit as st
import pickle
import numpy as np


# Load model
model = pickle.load(open("car_price_model.pkl","rb"))
fuel_encoder = pickle.load(open("fuel_encoder.pkl","rb"))
trans_encoder = pickle.load(open("trans_encoder.pkl","rb"))


st.title("🚗 Used Car Price Prediction")


st.write("Enter car details below")


# User Inputs
car_age = st.slider("Car Age (years)",0,20)

fuel_type = st.selectbox("Fuel Type",fuel_encoder.classes_)

transmission = st.selectbox("Transmission",trans_encoder.classes_)

owner = st.selectbox("Owner Type",[0,1,2,3])

kms_driven = st.number_input("Kilometers Driven",0,300000)


# Encode categorical inputs
fuel_encoded = fuel_encoder.transform([fuel_type])[0]
trans_encoded = trans_encoder.transform([transmission])[0]


# Prediction
if st.button("Predict Price"):

    input_data = np.array([[car_age,fuel_encoded,trans_encoded,owner,kms_driven]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Car Price: ₹ {int(prediction[0])}")