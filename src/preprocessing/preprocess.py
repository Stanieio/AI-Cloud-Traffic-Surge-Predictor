import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_preprocess(filepath):
    # Load CSV
    df = pd.read_csv(filepath)

    print("Original Data:")
    print(df.head())

    # Sort by hour (important for time series)
    df = df.sort_values('hour')

    # Normalize traffic (0 to 1)
    scaler = MinMaxScaler()
    df['scaled_traffic'] = scaler.fit_transform(df[['traffic']])

    print("\nAfter Scaling:")
    print(df.head())

    return df, scaler


def save_processed_data(df):
    # Create processed folder if not exists
    os.makedirs("data/processed", exist_ok=True)

    # Save processed file
    df.to_csv("data/processed/traffic_processed.csv", index=False)
    print("\nProcessed file saved in data/processed/")


def plot_traffic(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['hour'], df['traffic'])
    plt.title("Traffic Pattern Over Time")
    plt.xlabel("Hour")
    plt.ylabel("Traffic")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "data/raw/traffic.csv"

    df, scaler = load_and_preprocess(file_path)
    save_processed_data(df)
    plot_traffic(df)
