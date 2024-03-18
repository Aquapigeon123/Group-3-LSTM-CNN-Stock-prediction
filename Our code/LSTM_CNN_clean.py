# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv1D, Flatten, MaxPooling1D, Bidirectional, LSTM, Dropout, TimeDistributed, MaxPool2D
import matplotlib.pyplot as plt

# Load the data
stock = pd.read_csv('../Data/AAPL.csv')

# Parameters
window_size = 50
num_features = 1

# Prepare input and output data
X = [np.array([(stock.iloc[i + j, 4] - stock["Close"][i]) / stock["Close"][i] for j in range(window_size)]).reshape(window_size, 1) for i in range(len(stock) - window_size - 1)]
Y = [np.array([(stock.iloc[i + window_size, 4] - stock["Close"][i]) / stock["Close"][i]]).reshape(1, 1) for i in range(len(stock) - window_size - 1)]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

# Number of training samples
n_train = len(X_train)

# Convert the data to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Reshape the data
X_train = X_train.reshape(X_train.shape[0], 1, 50, 1)
X_test = X_test.reshape(X_test.shape[0], 1, 50, 1)

# Create the model using Sequential API
model = Sequential()

# Create the model using Sequential API
model = Sequential()

# Conv 1D layer with specified input shape
model.add(TimeDistributed(Conv1D(filters=128, kernel_size=7, activation="relu", input_shape=(None,window_size, num_features))))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

# Conv 1D layer
model.add(TimeDistributed(Conv1D(filters=256, kernel_size=5, activation="relu")))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

# Conv 1D layer
model.add(TimeDistributed(Conv1D(filters=512, kernel_size=3, activation="relu")))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

# Flatten layer
model.add(TimeDistributed(Flatten()))

# Bidirectional LSTM layer
model.add(Bidirectional(LSTM(units=200, return_sequences=True)))
model.add(Dropout(0.5))

# Bidirectional LSTM layer
model.add(Bidirectional(LSTM(units=200)))
model.add(Dropout(0.5))

# Dense layer for single output (predicted price)
model.add(Dense(1, activation="linear"))

# Compile the model for regression task (mse loss, mae metric)
model.compile(optimizer='RMSprop', loss='mse')

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=64, shuffle=False)

vall_loss = model.history.history['val_loss']

# Predict the stock price
predicted = model.predict(X_test)
predicted = np.array(predicted[:, 0]).reshape(-1, 1)

Y_test = Y_test[:, 0]

# Convert predicted and test_label back to original scale
temp = stock["Close"][n_train:n_train+len(X_test)]
Y_test = np.multiply(Y_test.flatten(), temp) + np.array(temp)
predicted = np.multiply(predicted.flatten(), temp) + np.array(temp)
    

# Plot the results
plt.plot(Y_test, color='black', label='Stock Price')
plt.plot(predicted, color='green', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

df = pd.DataFrame({"Epochs": range(1, 21),
                   "Validation Loss": vall_loss})
