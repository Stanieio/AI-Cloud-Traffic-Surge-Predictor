import pandas as pd
import matplotlib.pyplot as plt

actual = pd.read_csv("data/raw/traffic.csv")
predicted = pd.read_csv("data/predictions.csv")

plt.figure(figsize=(12,6))
plt.plot(actual["traffic"], label="Actual Traffic")
plt.plot(predicted["predicted_traffic"], label="Predicted Traffic")

plt.title("Traffic Forecast vs Actual")
plt.xlabel("Time")
plt.ylabel("Traffic")
plt.legend()

plt.savefig("forecast.png")
plt.show()
