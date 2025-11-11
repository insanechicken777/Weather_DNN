# ==============================================================================
# Step 1: Import all necessary libraries
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Model Specific Import ---
from sklearn.linear_model import LinearRegression

print("Libraries imported successfully for MRA.")

# ==============================================================================
# Step 2: Load and Prepare the Real Weather Data
# ==============================================================================
file_path = 'kma_weather_data.csv'
# (All code from your Step 2 and 2.5 is identical)
# ... (rest of your data loading) ...

# --- [YOUR DATA LOADING & FABRICATION CODE FROM railway.py GOES HERE] ---
# ... (Assuming your data loading, feature engineering, and 
# ...  target fabrication code from Step 2 & 2.5 is here) ...
#
# --- [START OF YOUR CODE] ---
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
# Step 2.5: Fabricate Realistic Data (Feature Engineering & Rule-Based Target)
# ==============================================================================
print("\nFabricating a richer, more realistic dataset...")
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
print("New features (season, time, day, railway type) created.")
np.random.seed(42)
base_risk = 0.05
risk_score = base_risk
risk_score += (df['Deepest snow cover on a day'] > 0) * 1.5
risk_score += (df['season'] == 'Winter') * 0.5
risk_score += (df['Daily precipitation'] > 30) * 0.4
risk_score += (df['Mean wind speed'] > 15) * 0.3
risk_score += (df['accident_time'].isin(['Morning Rush', 'Evening Rush'])) * 0.2
risk_score += (df['railway_classification'] == 'Main Line') * 0.1
y = pd.Series(np.random.poisson(lam=risk_score), name='Casualties')
print("Rule-based casualties generated.")
numerical_features = [
    'Mean temperature', 'Daily precipitation', 'Mean wind speed',
    'Mean relative humidity', 'Deepest snow cover on a day',
    'Deepest snowfall on the day', 'Mean ground temperature'
]
categorical_features = ['day_of_week', 'season', 'accident_time', 'railway_classification']
X = df[numerical_features + categorical_features]
print("\nFirst 5 rows of the new, richer input data (X):")
print(X.head())
# --- [END OF YOUR CODE] ---


# ==============================================================================
# Step 3: Data Preprocessing for the Model
# ==============================================================================
# (This step is IDENTICAL to your railway.py)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Handle NaNs from the original data (e.g., 'Daily precipitation')
X_train_processed[np.isnan(X_train_processed)] = 0
X_test_processed[np.isnan(X_test_processed)] = 0

print(f"\nData preprocessed. Input shape after one-hot encoding: {X_train_processed.shape}")
print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

# ==============================================================================
# Step 4: Train the MRA (Linear Regression) Model
# ==============================================================================
print("\n--- Training MRA Model ---")
model = LinearRegression()

print("Starting model training...")
model.fit(X_train_processed, y_train)
print("Model training finished.")

# ==============================================================================
# Step 5: Evaluate the MRA Model
# ==============================================================================
print("\n--- Evaluating MRA Model ---")
print("Evaluating model on the test data...")
predictions = model.predict(X_test_processed)

# Calculate metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions)) # RMSE is sqrt of MSE

print(f"Test MAE (Mean Absolute Error): {mae:.4f}")
print(f"Test RMSE (Root Mean Squared Error): {rmse:.4f}")

print("\n--- Generating and Displaying Predictions ---")
results_df = pd.DataFrame({
    'Actual Casualties': y_test.values,
    'Predicted Casualties': np.round(predictions).astype(int)
})
# Ensure predictions are not negative
results_df['Predicted Casualties'] = results_df['Predicted Casualties'].apply(lambda x: max(0, x))

print("\nSide-by-side comparison of Actual vs. Predicted Casualties (first 20):")
print(results_df.head(20))

print("\nProcess finished.")