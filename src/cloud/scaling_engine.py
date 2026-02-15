import pandas as pd

# Load predictions
predicted = pd.read_csv("data/predictions.csv")

# Define capacity levels
LOW_CAPACITY = 2      # 2 servers
MEDIUM_CAPACITY = 5   # 5 servers
HIGH_CAPACITY = 10    # 10 servers

print("\n☁️ CLOUD SCALING ENGINE\n")

for i, value in enumerate(predicted["predicted_traffic"]):

    if value < 200:
        servers = LOW_CAPACITY
        level = "LOW"

    elif 200 <= value < 400:
        servers = MEDIUM_CAPACITY
        level = "MEDIUM"

    else:
        servers = HIGH_CAPACITY
        level = "HIGH"

    print(f"Time {i} | Predicted Traffic: {value:.2f} → Scaling Level: {level} | Servers: {servers}")

print("\n✅ Auto-scaling simulation completed")

