import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data = pd.read_csv("data/raw/traffic.csv")

traffic = data["traffic"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
traffic_scaled = scaler.fit_transform(traffic)

# Create sequences
X = []
y = []

sequence_length = 10

for i in range(sequence_length, len(traffic_scaled)):
    X.append(traffic_scaled[i-sequence_length:i, 0])
    y.append(traffic_scaled[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=10, batch_size=16)

# Save model
model.save("src/model/lstm_model.h5")

# Predict on training data
predictions = model.predict(X)

# Convert back to original scale
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
actual = scaler.inverse_transform(y.reshape(-1,1))

# Plot results
plt.plot(actual, label="Actual Traffic")
plt.plot(predictions, label="Predicted Traffic")
plt.legend()
plt.title("LSTM Traffic Prediction")
plt.show()

print("Model training completed!")
