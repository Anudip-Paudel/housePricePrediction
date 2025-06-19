import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the price scaler (if you want to inverse-transform predictions)
with open("price_scaler.pkl", "rb") as f:
    price_scaler = pickle.load(f)

st.title("🏠 House Price Prediction")

# Reusable Yes/No selector
def yes_no_to_binary(label):
    return 1 if st.selectbox(label, ['Yes', 'No']) == 'Yes' else 0

# User Inputs
area = st.number_input("Area (sqft)", value=3000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=1)
mainroad = yes_no_to_binary("Main Road Access")
guestroom = yes_no_to_binary("Guest Room")
basement = yes_no_to_binary("Basement")
hotwaterheating = yes_no_to_binary("Hot Water Heating")
airconditioning = yes_no_to_binary("Air Conditioning")
parking = st.number_input("Parking Spaces", value=1)
prefarea = yes_no_to_binary("Preferred Area")

# Furnishing status (Ordinal Encoding)
furnishing = st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])
furnishing_map = {
    "unfurnished": 0,
    "semi-furnished": 1,
    "furnished": 2
}
furnishing_encoded = furnishing_map[furnishing]

# Build input DataFrame
input_data = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishing_encoded
}])

# Predict
if st.button("Predict Price"):
    scaled_price = model.predict(input_data)[0]
    actual_price = price_scaler.inverse_transform([[scaled_price]])[0][0]
    st.success(f"🏷️ Estimated House Price: ₹ {int(actual_price):,}")
