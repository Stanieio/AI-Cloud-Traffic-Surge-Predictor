import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==============================
# 1️⃣ Load Trained Model
# ==============================

model = load_model("src/model/lstm_model.h5", compile=False)

# ==============================
# 2️⃣ Load Dataset
# ==============================

data = pd.read_csv("data/raw/traffic.csv")

traffic = data["traffic"].values.reshape(-1, 1)

# ==============================
# 3️⃣ Scale Data
# ==============================

scaler = MinMaxScaler()
traffic_scaled = scaler.fit_transform(traffic)

# ==============================
# 4️⃣ Create Sequences
# ==============================

X = []
window_size = 10

for i in range(window_size, len(traffic_scaled)):
    X.append(traffic_scaled[i - window_size:i])

X = np.array(X)

# ==============================
# 5️⃣ Make Predictions
# ==============================

predictions_scaled = model.predict(X)
predictions = scaler.inverse_transform(predictions_scaled)

# ==============================
# 6️⃣ Save Predictions
# ==============================

pd.DataFrame(predictions, columns=["predicted_traffic"]).to_csv(
    "data/predictions.csv", index=False
)

print("✅ Predictions saved to data/predictions.csv")

# ==============================
# 7️⃣ Model Evaluation (Professional Way)
# ==============================

actual_values = traffic[window_size:]

mse = mean_squared_error(actual_values, predictions)
mae = mean_absolute_error(actual_values, predictions)

print("\n📊 Model Evaluation Results")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

print("\n🚀 Prediction & Evaluation Completed Successfully!")
