import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('Housing.csv')

    # Remove outliers
    z_scores = np.abs(stats.zscore(data["price"]))
    data = data[z_scores < 3]

    # Scale price
    scaler = MinMaxScaler()
    data['price'] = scaler.fit_transform(data[['price']])

    # Encode binary columns
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        data[col] = data[col].map({'yes': 1, 'no': 0})

    # Encode ordinal column
    ordinal_encoder = OrdinalEncoder(categories=[['furnished', 'semi-furnished', 'unfurnished']])
    data["furnishingstatus"] = ordinal_encoder.fit_transform(data[['furnishingstatus']])

    return data

# Train model
@st.cache_resource
def train_model(data):
    y = data['price']
    X = data.drop(['price'], axis=1)
    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns

# Streamlit UI
def main():
    st.title("🏡 House Price Predictor")
    data = load_data()
    model, feature_names = train_model(data)

    st.subheader("Enter House Features:")

    user_input = {}
    for col in feature_names:
        if col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
            user_input[col] = st.selectbox(col, ['yes', 'no'])
        elif col == 'furnishingstatus':
            user_input[col] = st.selectbox(col, ['furnished', 'semi-furnished', 'unfurnished'])
        else:
            user_input[col] = st.number_input(col, min_value=0)

    # Preprocess user input
    input_df = pd.DataFrame([user_input])
    for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
        input_df[col] = input_df[col].map({'yes': 1, 'no': 0})
    ord_enc = OrdinalEncoder(categories=[['furnished', 'semi-furnished', 'unfurnished']])
    input_df['furnishingstatus'] = ord_enc.fit_transform(input_df[['furnishingstatus']])

    # Predict normalized price
    pred_price = model.predict(input_df)[0]

    # Convert to actual price
    scaler = MinMaxScaler()
    scaler.fit(pd.read_csv('Housing.csv')[['price']])  # Load original data for inverse scaling
    actual_price = scaler.inverse_transform([[pred_price]])[0][0]

    # Show result
    st.success(f"🏷️ Predicted House Price: ₹ {actual_price:,.2f}")

    # Optional visualizations
    if st.checkbox("📊 Show Correlation Heatmap"):
        data_copy = data.copy()
        data_copy.drop(['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                        'airconditioning', 'prefarea'], axis=1, inplace=True)
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(data_copy.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
