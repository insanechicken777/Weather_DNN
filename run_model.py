# ==============================================================================
# Step 1: Import all necessary libraries
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import os
import seaborn as sns

print("Libraries imported successfully.")

# ==============================================================================
# Step 2: Load and Prepare the Real Weather Data
# ==============================================================================
file_path = 'kma_weather_data.csv'
# ... (The data loading part remains the same) ...
column_names = [
    'Station ID', 'Station Name', 'Timestamp', 'Mean temperature',
    'Daily precipitation', 'Mean wind speed', 'Mean relative humidity',
    'Deepest snow cover on a day', 'Deepest snowfall on the day',
    'Phenomenon number', 'Mean ground temperature', '5cm ground temperature',
    '10cm ground temperature', '20cm ground temperature', '30cm ground temperature'
]
try:
    df = pd.read_csv(
        file_path, encoding='euc-kr', header=None,
        skiprows=1, names=column_names
    )
    print(f"Successfully loaded data from {file_path}.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# ==============================================================================
# Step 2.5: Fabricate Realistic Data with Advanced Feature Engineering
# ==============================================================================
print("\nFabricating a richer, more realistic dataset...")

# --- Feature Engineering ---
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
df['day_of_week'] = df['Timestamp'].dt.day_name()
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'
df['season'] = df['Timestamp'].dt.month.apply(get_season)
def get_accident_time(hour):
    if 6 <= hour < 10: return 'Morning Rush'
    elif 10 <= hour < 16: return 'Midday'
    elif 16 <= hour < 20: return 'Evening Rush'
    else: return 'Night'
df['accident_time'] = df['Timestamp'].dt.hour.apply(get_accident_time)
railway_types = ['Main Line', 'Branch Line', 'Metro', 'Freight Line']
df['railway_classification'] = np.random.choice(railway_types, size=len(df), p=[0.4, 0.3, 0.2, 0.1])
df.fillna(0, inplace=True) # Fill NaNs before calculations

# --- NEW: Add the wind_chill feature ---
df['wind_chill'] = df['Mean temperature'] - (df['Mean wind speed'] * 2)
print("New features (season, time, day, railway type, wind_chill) created.")

# --- Rule-Based Casualty Generation (with DRAMATICALLY STRONGER signals) ---
np.random.seed(42)
base_risk = 0.1
risk_score = base_risk
risk_score += (df['Deepest snow cover on a day'] > 0) * 8.0
risk_score += (df['season'] == 'Winter') * 1.5
risk_score += (df['Daily precipitation'] > 30) * 4.0
risk_score += (df['Mean wind speed'] > 15) * 2.0
y = pd.Series(np.random.poisson(lam=risk_score), name='Casualties')
print("Rule-based casualties generated for REGRESSION task.")

# Define our final list of features for the model
numerical_features = [
    'Mean temperature', 'Daily precipitation', 'Mean wind speed',
    'Mean relative humidity', 'Deepest snow cover on a day', 'wind_chill' # Added wind_chill
]
categorical_features = ['day_of_week', 'season', 'accident_time', 'railway_classification']
X = df[numerical_features + categorical_features]

# ==============================================================================
# Step 3: Data Preprocessing for the Model
# ==============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"\nData preprocessed. Input shape: {X_train_processed.shape}")

# ==============================================================================
# Step 4: Build and Train a REGRESSION Model
# ==============================================================================
MODEL_FILENAME = 'weather_dnn_regression_v3.keras' # New model name

if os.path.exists(MODEL_FILENAME):
    print(f"\nFound saved model: {MODEL_FILENAME}. Loading pre-trained model...")
    model = tf.keras.models.load_model(MODEL_FILENAME)
else:
    print(f"\nNo saved model found. Training a new REGRESSION model...")
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        ### REVERTED: The output layer for regression (no activation)
        Dense(1) 
    ])

    ### REVERTED: Compile the model for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    print("\nStarting model training...")
    history = model.fit(
        X_train_processed, y_train,
        epochs=150,
        batch_size=64,
        validation_data=(X_test_processed, y_test)
    )
    #pd.DataFrame(history.history).to_csv('history_without_dropout.csv', index=False)
    model.save(MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}.")

# ==============================================================================
# Step 5: Evaluate the REGRESSION Model
# ==============================================================================
print("\nEvaluating model on the test data...")
loss, mae = model.evaluate(X_test_processed, y_test, verbose=0)
rmse = np.sqrt(loss)

print(f"Test MAE (Mean Absolute Error): {mae:.4f}")
print(f"Test RMSE (Root Mean Squared Error): {rmse:.4f}")

# --- Generating and Displaying Predictions ---
print("\n--- Generating and Displaying Predictions ---")
predictions = model.predict(X_test_processed).flatten()
results_df = pd.DataFrame({
    'Actual Casualties': y_test.values,
    'Predicted Casualties': np.round(predictions).astype(int)
})
results_df['Predicted Casualties'] = results_df['Predicted Casualties'].apply(lambda x: max(0, x))
print("\nSide-by-side comparison of Actual vs. Predicted Casualties (first 20):")
print(results_df.head(20))

# --- Checking predictions for high-risk scenarios ---
print("\n--- Checking predictions for high-risk scenarios ---")
full_test_df = X_test.copy()
full_test_df['Actual Casualties'] = y_test
full_test_df['Predicted Casualties'] = results_df['Predicted Casualties'].values # Use .values for safe assignment
risky_days = full_test_df[full_test_df['Deepest snow cover on a day'] > 0]
print("\nPredictions for days with snow:")
print(risky_days[['Actual Casualties', 'Predicted Casualties']].head(20))

# ==============================================================================
# Step 6: Visual Evaluation of the Model's Performance
# ==============================================================================
print("\n--- Generating Visualizations for Model Evaluation ---")

# --- Plot 1: Actual vs. Predicted Scatter Plot ---
plt.figure(figsize=(10, 8))
# We use seaborn (sns) for better aesthetics
sns.scatterplot(x='Actual Casualties', y='Predicted Casualties', data=results_df, alpha=0.5)
max_val = max(results_df['Actual Casualties'].max(), results_df['Predicted Casualties'].max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', lw=2, label='Perfect Prediction')
plt.title('Actual vs. Predicted Casualties', fontsize=16)
plt.xlabel('Actual Casualties', fontsize=12)
plt.ylabel('Predicted Casualties', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# --- Plot 2: Residuals Plot ---
results_df['Residuals'] = results_df['Actual Casualties'] - results_df['Predicted Casualties']
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Predicted Casualties', y='Residuals', data=results_df, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', lw=2)
plt.title('Residuals vs. Predicted Values', fontsize=16)
plt.xlabel('Predicted Casualties', fontsize=12)
plt.ylabel('Residuals (Error)', fontsize=12)
plt.grid(True)
plt.show()

# --- Plot 3: Distribution of Errors Histogram ---
plt.figure(figsize=(10, 8))
sns.histplot(results_df['Residuals'], bins=30, kde=True)
plt.title('Distribution of Prediction Errors (Residuals)', fontsize=16)
plt.xlabel('Error (Actual - Predicted)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)
plt.show()

print("\nProcess finished.")