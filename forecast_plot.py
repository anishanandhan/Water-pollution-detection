import matplotlib.pyplot as plt
import numpy as np
import os

# Dummy LSTM prediction (replace with your actual model predictions)
hours = np.arange(24)
predictions = np.sin(hours / 3) + np.random.normal(0, 0.2, 24) + 5  # Just for demo

# Plot
plt.figure(figsize=(10, 4))
plt.plot(hours, predictions, marker='o', linestyle='-', color='blue')
plt.title("Turbidity Forecast (Next 24 Hours)")
plt.xlabel("Hour from Now")
plt.ylabel("Turbidity (NTU)")
plt.grid(True)
plt.tight_layout()

# Save to static folder
os.makedirs("static", exist_ok=True)
plt.savefig("static/turbidity_forecast.png")
plt.close()
