import pandas as pd

# Load predictions
predicted = pd.read_csv("data/predictions.csv")

# You can tune this
THRESHOLD_HIGH = 400
THRESHOLD_CRITICAL = 480

print("\n🚨 ALERT SYSTEM STATUS\n")

for i, value in enumerate(predicted["predicted_traffic"]):

    if value >= THRESHOLD_CRITICAL:
        print(f"🔥 CRITICAL SURGE at time {i} | Traffic: {value:.2f}")
    
    elif value >= THRESHOLD_HIGH:
        print(f"⚠ HIGH TRAFFIC at time {i} | Traffic: {value:.2f}")

print("\n✅ Alert scanning completed")
