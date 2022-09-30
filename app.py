import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import load_model



# Load data

start='2012-08-01'
end='2021-07-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter stock name ','SBIN.NS')
data = web.DataReader(user_input, 'yahoo', start, end)


# Discribing Data
st.subheader('Date from 2012 - 2021')
st.write(data.describe())


# Data visualization
st.subheader('Closing price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close,'b',label='closing price of the company')
plt.xlabel('time',fontsize=18)
plt.ylabel('share price of the company',fontsize=18)
plt.legend()
st.pyplot(fig)



st.subheader('Closing price vs Time chart with 100day MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'g',label='100day MA')
plt.plot(data.Close,'b',label='closing price of the company')
plt.xlabel('time',fontsize=18)
plt.ylabel('share price of the company',fontsize=18)
plt.legend()
st.pyplot(fig)



st.subheader('Closing price vs Time chart with 100day MA and 200day MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'g',label='100day MA')
plt.plot(ma200,'r',label='200day MA')
plt.plot(data.Close,'b',label='closing price of the company')
plt.xlabel('time',fontsize=18)
plt.ylabel('share price of the company',fontsize=18)
plt.legend()
st.pyplot(fig)



# prepare data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))


# Loading my model
model = load_model('keras_model.h5')



# load test data

test_start = '2021-07-31'
test_end = '2022-08-10'
test_data = web.DataReader(user_input, data_source='yahoo',start=test_start, end=test_end)
actual_prices = test_data['Close'].values


total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

prediction_days = 60
model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# make prediction on test data

x_test = []

for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i - prediction_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)



# plot the test predictions

st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='b',label='Actual price of the company')
plt.plot(predicted_prices, color='black',label='Predicted price of the company')
plt.title('company share price chart')
plt.xlabel('time')
plt.ylabel('share price of the company')
plt.legend()
st.pyplot(fig2)



# predict next day price

real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0],real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")