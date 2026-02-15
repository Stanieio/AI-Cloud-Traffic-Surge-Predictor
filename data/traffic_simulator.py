import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate 7 days of minute-level data
minutes = 7 * 24 * 60
start_time = datetime.now()

timestamps = [start_time + timedelta(minutes=i) for i in range(minutes)]

# Base traffic pattern (normal daily cycle)
base_traffic = 100 + 50 * np.sin(np.linspace(0, 20 * np.pi, minutes))

# Random noise
noise = np.random.normal(0, 10, minutes)

# Simulated spikes (events)
spikes = np.zeros(minutes)
for _ in range(10):
    spike_start = np.random.randint(0, minutes-60)
    spikes[spike_start:spike_start+30] += np.random.randint(200, 400)

# Final traffic
requests = base_traffic + noise + spikes

cpu_usage = requests * 0.05 + np.random.normal(0, 5, minutes)
memory_usage = requests * 0.03 + np.random.normal(0, 3, minutes)

df = pd.DataFrame({
    "timestamp": timestamps,
    "requests": requests,
    "cpu_usage": cpu_usage,
    "memory_usage": memory_usage
})

df.to_csv("data/traffic_data.csv", index=False)

print("Traffic data generated successfully!")


plt.figure(figsize=(15,5))
plt.plot(df["timestamp"], df["requests"])
plt.title("Simulated Cloud Traffic")
plt.xlabel("Time")
plt.ylabel("Requests")
plt.show()
