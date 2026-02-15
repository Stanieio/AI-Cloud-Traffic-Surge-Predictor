import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

hours = 24 * 45  # 45 days hourly data
time = np.arange(hours)

# Daily pattern
daily_pattern = 120 + 50 * np.sin(2 * np.pi * time / 24)

# Weekly pattern
weekly_pattern = 30 * np.sin(2 * np.pi * time / (24 * 7))

# Random noise
noise = np.random.normal(0, 15, hours)

# Sale spikes (long duration)
sale = np.zeros(hours)
sale_days = np.random.choice(range(24, hours-24), 4)
for day in sale_days:
    sale[day:day+10] += np.random.randint(200, 350)

# Sudden attack spikes
attack = np.zeros(hours)
attack_indices = np.random.choice(hours, 6)
attack[attack_indices] = np.random.randint(300, 500)

traffic = daily_pattern + weekly_pattern + noise + sale + attack

df = pd.DataFrame({
    "hour": time,
    "traffic": traffic
})

df.to_csv("data/raw/traffic.csv", index=False)

plt.figure(figsize=(14,5))
plt.plot(df["traffic"])
plt.title("Simulated Cloud Traffic (45 Days)")
plt.xlabel("Hour")
plt.ylabel("Requests")
plt.show()
