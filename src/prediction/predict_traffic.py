import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def predict_traffic():

    # Load model
    model = load_model("src/model/lstm_model.h5", compile=False)

    # Load data
    data = pd.read_csv("data/processed/traffic_processed.csv")

    traffic = data["traffic"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    traffic_scaled = scaler.fit_transform(traffic)

    sequence_length = 60

    last_sequence = traffic_scaled[-sequence_length:]
    last_sequence = last_sequence.reshape(1, sequence_length, 1)

    predicted_scaled = model.predict(last_sequence)

    predicted_value = scaler.inverse_transform(predicted_scaled)

    final_value = float(predicted_value[0][0])

    print("Predicted Traffic:", final_value)

    return final_value
