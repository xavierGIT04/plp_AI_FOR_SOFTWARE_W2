import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_absolute_error


# --- 1. CONFIGURATION ---
FILE_NAME = 'SDG13_Climate_Action_Data.csv'
FEATURES = [
    'CO2_Metric_Tons_Per_Capita',
    'Gas consumption per capita(mł)',
    'Coal consumption per capita(Ton)',
    'Oil consumption per capita(mł)',
    'Population'
]
TARGET_COLUMN_INDEX = 0  # Index of 'CO2_Metric_Tons_Per_Capita' in the FEATURES list
LOOK_BACK = 10  # Number of previous years to use for prediction
TRAIN_SPLIT_RATIO = 0.8
EPOCHS = 100
BATCH_SIZE = 32

# --- 2. DATA PREPARATION ---

# Load Data and Initial Cleaning
df = pd.read_csv(FILE_NAME)

# Handle Missing Data using forward-fill for time-series, then fill remaining with 0
df = df.ffill()
df = df.fillna(0)

# Select numerical data and convert to NumPy array
data = df[FEATURES].values

# Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Chronological Split
training_data_len = int(np.ceil(len(scaled_data) * TRAIN_SPLIT_RATIO))
train_data = scaled_data[0:training_data_len, :]
test_data = scaled_data[training_data_len:, :]


# --- 3. SEQUENCE CREATION FUNCTION ---

def create_sequences(dataset, look_back):
    """
    Transforms the time-series data into sequences (X) and next-step targets (Y)
    required for LSTM training.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # X: Features from the last 'look_back' years (all columns)
        X.append(dataset[i:(i + look_back), :])

        # Y: Target (CO2, which is index 0) for the next year
        Y.append(dataset[i + look_back, TARGET_COLUMN_INDEX])
    return np.array(X), np.array(Y)


# Apply Sequence Creation
X_train, Y_train = create_sequences(train_data, LOOK_BACK)
X_test, Y_test = create_sequences(test_data, LOOK_BACK)

# --- 4. MODEL DEFINITION ---

# Build the LSTM Model (Dynamic Neural Network)
model = Sequential()
# Input layer (with return_sequences=True for stacked LSTMs)
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Second LSTM layer (return_sequences=False for output to dense layer)
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output layer (1 unit for regression)
model.add(Dense(units=1, activation='linear'))

# Compile Model
model.compile(optimizer='adam', loss='mae')
print("--- Model Summary ---")
model.summary()

# --- 5. TRAINING AND EVALUATION ---

print("\n--- Model Training ---")
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, Y_test),
    verbose=0
)

# Make Predictions
predictions_scaled = model.predict(X_test)

# Inverse Transform (Crucial Step!)
# The scaler was fitted on all 5 features. To inverse transform the prediction,
# we must create a dummy array with the prediction in the CO2 column (index 0).
predictions_full = np.zeros((len(predictions_scaled), len(FEATURES)))
predictions_full[:, TARGET_COLUMN_INDEX] = predictions_scaled.flatten()

actual_full = np.zeros((len(Y_test), len(FEATURES)))
actual_full[:, TARGET_COLUMN_INDEX] = Y_test.flatten()

predictions = scaler.inverse_transform(predictions_full)[:, TARGET_COLUMN_INDEX]
actual = scaler.inverse_transform(actual_full)[:, TARGET_COLUMN_INDEX]

# Evaluate and Print MAE
mae = mean_absolute_error(actual, predictions)
print(f"\n--- Final Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.3f} Metric Tons Per Capita")
print(f"Interpretation: On average, the model's CO2 prediction is off by {mae:.3f} metric tons per capita.")

# --- 6. VISUALIZATION (FOR DEMO/REPORT) ---

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(actual, label='Actual CO2 Emissions', color='blue')
plt.plot(predictions, label='Predicted CO2 Emissions', color='red', linestyle='--')
plt.title(f'CO2 Emissions Forecasting (LSTM) | MAE: {mae:.3f}')
plt.xlabel(f'Time Steps in Test Set (Look Back = {LOOK_BACK})')
plt.ylabel('CO2 Metric Tons Per Capita')
plt.legend()
plt.grid(True)

# Save the plot for your report/presentation
plt.savefig('co2_forecasting_results.png')
plt.show()
print("Results saved to 'co2_forecasting_results.png'")