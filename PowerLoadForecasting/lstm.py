# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:53:03 2024

@author: qing
"""

import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Load csv
df = pd.read_csv('C:/Users/qing/Desktop/全部数据/大工业用电最大值.CSV', parse_dates=['date'])
print(df)
def load_data(start, end):

    dataframe = df.copy()
    #dataframe = dataframe.loc[dataframe.Name == company, :]
    dataframe = dataframe.loc[(dataframe['date'] > start) & (dataframe['date'] < end), :]
    dataframe = dataframe.rename(columns = {'kw': 'PowerLoad'})
    return dataframe
#COMPANY = 'Accor'
START_DATE = dt.datetime(2019,1,1)
END_DATE = dt.datetime(2020,5,1)

START_DATE_TEST = END_DATE
data = load_data(
    #company = COMPANY,
                 start = START_DATE,
                 end = END_DATE)
# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['PowerLoad'].values.reshape(-1,1))
# Set the number of days used for prediction
prediction_days = 365

# Initialize empty lists for training data input and output
x_train = []
y_train = []

# Iterate through the scaled data, starting from the prediction_days index
for x in range(prediction_days, len(scaled_data)):
    # Append the previous 'prediction_days' values to x_train
    x_train.append(scaled_data[x - prediction_days:x, 0])
    # Append the current value to y_train
    y_train.append(scaled_data[x, 0])

# Convert the x_train and y_train lists to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train to a 3D array with the appropriate dimensions for the LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


def LSTM_model():
    """
    Create and configure an LSTM model for power load prediction.

    :return: The configured LSTM model (keras.Sequential)
    """

    # Initialize a sequential model
    model = Sequential()

    # Add the first LSTM layer with 50 units, input shape, and return sequences
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a second LSTM layer with 50 units and return sequences
    model.add(LSTM(units=50, return_sequences=True))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a third LSTM layer with 50 units
    model.add(LSTM(units=50))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a dense output layer with one unit
    model.add(Dense(units=1))

    return model


model = LSTM_model()
model.summary()
model.compile(
    optimizer='Adam', 
    loss='mean_squared_error'
)

# Define callbacks

# Save weights only for best model
checkpointer = ModelCheckpoint(
    filepath = 'weights_best.hdf5', 
    verbose = 2, 
    save_best_only = True
)

model.fit(
    x_train, 
    y_train, 
    epochs=25, 
    batch_size = 32,
    callbacks = [checkpointer]
)

# Load test data for the specified company and date range
test_data = load_data(
    start=START_DATE_TEST,
    end=dt.datetime.now()
)

# Extract the actual closing prices from the test data
actual_load = test_data['PowerLoad'].values

# Concatenate the training and test data along the 'Close' column
total_dataset = pd.concat((data['PowerLoad'], test_data['PowerLoad']), axis=0)

# Extract the relevant portion of the dataset for model inputs
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

# Reshape the model inputs to a 2D array with a single column
model_inputs = model_inputs.reshape(-1, 1)

# Apply the same scaling used for training data to the model inputs
model_inputs = scaler.transform(model_inputs)
# Initialize an empty list for test data input
x_test = []

# Iterate through the model inputs, starting from the prediction_days index
for x in range(prediction_days, len(model_inputs)):
    # Append the previous 'prediction_days' values to x_test
    x_test.append(model_inputs[x-prediction_days:x, 0])

# Convert the x_test list to a numpy array
x_test = np.array(x_test)

# Reshape x_test to a 3D array with the appropriate dimensions for the LSTM model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Generate price predictions using the LSTM model
predicted_load = model.predict(x_test)

# Invert the scaling applied to the predicted prices to obtain actual values
predicted_load = scaler.inverse_transform(predicted_load)
# Plot the actual prices using a black line
plt.plot(actual_load, color='black', label="Actual power load")

# Plot the predicted prices using a green line
plt.plot(predicted_load, color='green', label="Predicted power load")

# Set the title of the plot using the company name
plt.title("power load forecast")

# Set the x-axis label as 'time'
plt.xlabel("time")

# Set the y-axis label using the company name
plt.ylabel("power load")

# Display a legend to differentiate the actual and predicted prices
plt.legend()

# Show the plot on the screen
plt.show()


# Extract the last 'prediction_days' values from the model inputs
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]

# Convert the real_data list to a numpy array
real_data = np.array(real_data)

# Reshape real_data to a 3D array with the appropriate dimensions for the LSTM model
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

# Generate a prediction using the LSTM model with the real_data input
prediction = model.predict(real_data)

# Invert the scaling applied to the prediction to obtain the actual value
prediction = scaler.inverse_transform(prediction)
predicted_load = pd.DataFrame({'RISK':predicted_load})
predicted_load.to_csv('C:\/Users\qing\Desktop\全部数据/LSTM预测结果.csv')
# Print the prediction result to the console
#print(f"Prediction: {prediction[0][0]}")