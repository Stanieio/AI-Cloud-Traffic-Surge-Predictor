import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ==============================
# 1️⃣ Load Data
# ==============================

data = pd.read_csv("data/processed/traffic_processed.csv")
traffic = data["traffic"].values.reshape(-1, 1)

# ==============================
# 2️⃣ Scale Data
# ==============================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(traffic)

# Save scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.save")

# ==============================
# 3️⃣ Create Sequences
# ==============================

def create_sequences(data, sequence_length=24):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# ==============================
# 4️⃣ Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ==============================
# 5️⃣ Build LSTM Model
# ==============================

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# ==============================
# 6️⃣ Early Stopping
# ==============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==============================
# 7️⃣ Train Model
# ==============================

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ==============================
# 8️⃣ Save Model
# ==============================

model.save("model/lstm_model.h5")

print("✅ Model and scaler saved successfully!")

# ==============================
# 9️⃣ Plot Training Loss
# ==============================

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train', 'Validation'])
plt.show()
