import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model("src/model/lstm_model.h5")

data = pd.read_csv("data/processed/traffic_processed.csv")


traffic = data["traffic"].values.reshape(-1, 1)

scaler = MinMaxScaler()
traffic_scaled = scaler.fit_transform(traffic)

sequence_length = 60

last_sequence = traffic_scaled[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, 1)

predicted_scaled = model.predict(last_sequence)
predicted_value = scaler.inverse_transform(predicted_scaled)

print("Predicted Traffic:", predicted_value[0][0])

threshold = traffic.mean() * 1.5

if predicted_value[0][0] > threshold:
    print("⚠ TRAFFIC SURGE DETECTED!")
    print("🚀 Scale Up Servers!")
else:
    print("✅ Traffic Normal")


plt.figure(figsize=(10,5))
plt.plot(traffic[-100:], label="Recent Traffic")
plt.scatter(len(traffic[-100:]), predicted_value[0][0], 
            color="red", label="Predicted")

plt.legend()
plt.title("Traffic Forecast")
plt.show()
