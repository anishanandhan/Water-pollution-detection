import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("water_quality_12.csv")

# Handle missing time column
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
else:
    df['Date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')  # Changed 'H' to 'h'

# Features and target
features = ['pH', 'BOD_avg', 'DO_avg', 'Conductivity_avg']
target = 'Turbidity'

# Convert non-numeric values to NaN
df[features] = df[features].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in the features or target
df = df.dropna(subset=features + [target])

# Normalize
scaler_x = MinMaxScaler()
scaled_x = scaler_x.fit_transform(df[features])

scaler_y = MinMaxScaler()
scaled_y = scaler_y.fit_transform(df[[target]])

# Create sequences
def create_sequences(x, y, seq_length=24):
    X, Y = [], []
    for i in range(len(x) - seq_length):
        X.append(x[i:i + seq_length])
        Y.append(y[i + seq_length])
    return np.array(X), np.array(Y)

sequence_length = 24
X, y = create_sequences(scaled_x, scaled_y, sequence_length)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Predict future turbidity
n_future = 24
last_sequence = X[-1]
future_predictions = []

current_seq = last_sequence.copy()
for _ in range(n_future):
    next_pred = model.predict(np.expand_dims(current_seq, axis=0), verbose=0)[0]
    future_predictions.append(next_pred)

    # Use same feature pattern (replace last turbidity with predicted)
    next_input = np.append(current_seq[1:], [current_seq[-1]], axis=0)
    current_seq = next_input

# Inverse transform
future_predictions = scaler_y.inverse_transform(future_predictions)

# Output future turbidity
future_df = pd.DataFrame({
    'Predicted_Turbidity': future_predictions.flatten(),
    'Time': pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(hours=1), periods=n_future, freq='H')
})

# Save prediction CSV
future_df.to_csv("predicted_turbidity.csv", index=False)
print("Saved predicted_turbidity.csv")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['Date'][-50:], df[target][-50:], label="Historical Turbidity")
plt.plot(future_df['Time'], future_df['Predicted_Turbidity'], label="Predicted", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Turbidity")
plt.title("Turbidity Forecast using LSTM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
