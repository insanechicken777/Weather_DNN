<<<<<<< HEAD
# Predictive Deep Learning Model for Weather-Related Railway Incidents

This project implements a Deep Neural Network (DNN) to explore the potential of forecasting railway casualty risk based on historical meteorological data. The model is built using TensorFlow and Keras, and the entire data processing pipeline is handled with Pandas and Scikit-learn.

The project demonstrates a complete end-to-end machine learning workflow, from acquiring and cleaning real-world data to building, training, evaluating, and deploying a predictive model.

## Key Features

- **Data Handling:** Ingests and cleans complex, real-world data from the Korea Meteorological Administration (KMA), correctly handling specific character encodings (`euc-kr`) and messy headers.
- **Deep Learning Model:** Constructs a 3-layer Deep Neural Network using TensorFlow/Keras to learn non-linear patterns between weather features and incident risk.
- **Model Persistence:** Includes a feature to save a fully trained model to a file. On subsequent runs, the script loads the pre-trained model, bypassing the need for retraining and enabling near-instant predictions.
- **Data Preprocessing:** Utilizes Scikit-learn for essential preprocessing steps, including feature scaling (`StandardScaler`) and data splitting (`train_test_split`).
- **Evaluation:** Measures model performance using standard regression metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Technology Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning Utilities:** Scikit-learn
- **Plotting & Visualization:** Matplotlib

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-folder>
    ```
2.  **Set up a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.\.venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib
    ```
4.  **Add the data:**
    Place your downloaded KMA weather data file in the root of the project folder and ensure it is named `kma_weather_data.csv`.

5.  **Run the script:**
    ```bash
    python run_model.py
    ```
    - The first run will train the model from scratch and save it as `weather_dnn_model.keras`.
    - All subsequent runs will load the saved model and skip training.

## Project Status

**Proof of Concept:** The current implementation uses **placeholder (randomly generated) casualty data** for demonstration purposes. This allows the full pipeline to be tested and validated. To turn this into a real-world predictive tool, the placeholder `y` variable would need to be replaced with a real historical incident dataset.
=======
# Weather_DNN
A Deep Neural Network in Python to predict railway casualty risk from historical weather data. Features data cleaning, a Keras/TensorFlow model, and model persistence for efficient prediction.
>>>>>>> 8f53d58d25f60a537c1d75b057da9d0bf1d025c0
