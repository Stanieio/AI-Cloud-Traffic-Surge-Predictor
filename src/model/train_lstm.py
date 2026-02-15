import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/raw/traffic.csv")

traffic = data['traffic'].values.reshape(-1, 1)

# -----------------------------
# Scale Data
# -----------------------------
scaler = MinMaxScaler()
traffic_scaled = scaler.fit_transform(traffic)

# Save scaler for future use
joblib.dump(scaler, "src/model/scaler.save")

# -----------------------------
# Create Sequences
# -----------------------------
def create_sequences(data, seq_length=20):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(traffic_scaled)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# Build Model
# -----------------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -----------------------------
# Save Model
# -----------------------------
model.save("src/model/lstm_model.h5")

print("Model training complete!")

# -----------------------------
# Plot Loss Graph
# -----------------------------
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
