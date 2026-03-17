import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("crowd_data.csv")

print(data.head())

zones = data["Zone"].unique()

for zone in zones:

    zone_data = data[data["Zone"] == zone]

    plt.figure()

    plt.plot(zone_data["Time"],zone_data["Inside"])

    plt.title(f"Crowd Trend for {zone}")

    plt.xlabel("Time")
    plt.ylabel("People Inside")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()