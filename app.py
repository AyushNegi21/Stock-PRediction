import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


# Set date range
start = '2010-01-01'
end = '2025-12-31'

# Streamlit UI
st.title('📈 Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'TCS.NS').strip().upper()

# Fetch stock data
df = yf.download(user_input, start=start, end=end, auto_adjust=True)

# Handling empty DataFrame case
if df.empty:
    st.error(f"⚠️ Failed to fetch data for {user_input}. It may be delisted or incorrect.")
    st.stop()

# Display data summary
st.subheader('📊 Stock Data Summary')
st.write(df.describe())

# Visualization
st.subheader('📉 Closing Price vs Time')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label="Closing Price", color='black')
plt.legend()
st.pyplot(fig)

st.subheader('📊 Closing Price with 100 & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.Close, label="Closing Price", color='black')
ax.plot(ma100, label="100-Day MA", color='red')
ax.plot(ma200, label="200-Day MA", color='blue')
ax.legend()
st.pyplot(fig)

# Split data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

if data_training.empty:
    st.error("⚠️ Not enough data to train the model.")
    st.stop()

# Normalize training data
data_training_array = scaler.fit_transform(data_training)

# Check if model file exists
if not os.path.exists("stock_model.h5"):
    st.error("⚠️ Model file 'stock_model.h5' not found! Please place it in the same directory.")
    st.stop()

# Load the trained model
model = load_model("stock_model.h5")

# Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_predicted = model.predict(x_test)

# Rescale values
scale_factor = 1 / scaler.scale_[0]
y_predicted *= scale_factor
y_test *= scale_factor

# Plot results
st.subheader('📊 Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
