
# Step 1: Import libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os  ### NEW: Import the 'os' library to check for file existence

print("Libraries imported successfully.")


# Step 2: Load and Prepare the Real Weather Data

file_path = 'kma_weather_data.csv'

# Define the correct column names in English, in the correct order.
column_names = [
    'Station ID', 'Station Name', 'Timestamp', 'Mean temperature',
    'Daily precipitation', 'Mean wind speed', 'Mean relative humidity',
    'Deepest snow cover on a day', 'Deepest snowfall on the day',
    'Phenomenon number', 'Mean ground temperature', '5cm ground temperature',
    '10cm ground temperature', '20cm ground temperature', '30cm ground temperature'
]

try:
    # Load the CSV, telling pandas there's NO header and to use our defined names.
    df = pd.read_csv(
        file_path,
        encoding='euc-kr',
        header=None,
        skiprows=1,
        names=column_names
    )
    print(f"Successfully loaded data from {file_path} and applied manual column names.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Define the features to be used for training.
weather_features = [
    'Mean temperature', 'Daily precipitation', 'Mean wind speed',
    'Mean relative humidity', 'Deepest snow cover on a day',
    'Deepest snowfall on the day', 'Mean ground temperature'
]

# Fill empty cells in feature columns with 0.
df[weather_features] = df[weather_features].fillna(0)
print("Missing values in feature columns filled with 0.")

# Create the input (X) and placeholder output (y)
X = df[weather_features]
np.random.seed(42)
y = pd.Series(np.random.randint(0, 5, size=len(X)))
print("Input features (X) and placeholder output (y) created.")


# Step 3: Data Preprocessing for the Model

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nData has been scaled using StandardScaler.")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")


# Step 4: Load Pre-Trained Model or Train a New One

MODEL_FILENAME = 'weather_dnn_model.keras'

#Check if the model file already exists
if os.path.exists(MODEL_FILENAME):
    print(f"\nFound saved model: {MODEL_FILENAME}")
    print("Loading pre-trained model...")
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("Model loaded successfully.")
else:
    print(f"\nNo saved model found.")
    print("Training a new model from scratch...")
    
    # Define the DNN model structure
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("\nDNN Model architecture summary:")
    model.summary()

    # Train the model
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64, # Using an efficient batch size
        verbose=1,
        validation_split=0.1
    )
    print("Model training completed.")
    
    ### NEW: Save the newly trained model to a file
    model.save(MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME} for future use.")


# Step 5: Evaluate the Model and Show Results

print("\nEvaluating model on the test data...")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
rmse = np.sqrt(loss)

print(f"Test MAE (Mean Absolute Error): {mae:.4f}")
print(f"Test RMSE (Root Mean Squared Error): {rmse:.4f}")

# --- Generating and Displaying Predictions ---
print("\n--- Generating and Displaying Predictions ---")
predictions = model.predict(X_test).flatten()
results_df = pd.DataFrame({
    'Actual Casualties': y_test,
    'Predicted Casualties (Raw)': predictions
})
results_df['Predicted Casualties (Rounded)'] = np.round(results_df['Predicted Casualties (Raw)']).astype(int)
results_df['Predicted Casualties (Rounded)'] = results_df['Predicted Casualties (Rounded)'].apply(lambda x: max(0, x))
print("\nSide-by-side comparison of Actual vs. Predicted Casualties (first 20):")
print(results_df.head(20))

# --- Plotting the training history (only if we just trained the model) ---

if 'history' in locals():
    print("\nDisplaying training history plots...")
    plt.figure(figsize=(12, 5))
    
    # Plot MAE
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE Over Epochs')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE) Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

print("\nProcess finished.")