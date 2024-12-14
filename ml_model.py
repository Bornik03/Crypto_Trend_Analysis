import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading data
def load_data(filename):
    data = pd.read_csv(filename)
    data.columns = [
        "Date", "Open", "High", "Low", "Close", "Historical High Price", "Days_Since_High", "%_Diff_High",
        "Historical Low Price", "Days_Since_Low", "%_Diff_Low", "Future_High", "Future_Low",
        "% Difference from Future High", "% Difference from Future Low"
    ]
    return data

# Training the machine learning model using TensorFlow
def train_model(filename):
    # Loading and preprocessing the data
    data = load_data(filename)

    # Define feature columns and target columns
    feature_columns = [
        "Days_Since_High", "%_Diff_High",
        "Days_Since_Low", "%_Diff_Low"
    ]
    target_columns = [
        "% Difference from Future High", "% Difference from Future Low"
    ]

    # Drop rows with missing values
    data = data.dropna(subset=feature_columns + target_columns)

    # Split the data into features (X) and targets (y)
    X = data[feature_columns]
    y = data[target_columns]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Building the TensorFlow model with additional complexity
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)  # Output layer with two neurons for the two target variables
    ])

    # Compiling the model with Huber loss
    model.compile(optimizer='adam', loss='huber')

    # Training model
    model.fit(X_train, y_train, epochs=200, batch_size=8, validation_split=0.2)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = tf.keras.losses.MeanSquaredError()(y_test, y_pred).numpy()
    accuracy = 100 - mse  # Rough accuracy metric based on error reduction

    print(f"Model Mean Squared Error: {mse}")
    print(f"Model Accuracy (approx): {accuracy:.2f}%")

    # Save the model and scaler
    model.save("trained_model.keras")
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)
    return accuracy

# Predict outcomes with the trained model
def predict_outcomes(days_since_high, pct_diff_from_high, days_since_low, pct_diff_from_low):
    # Loading the model and scaler
    model = tf.keras.models.load_model("trained_model.keras")
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")

    # Prepare input data and scale it
    input_data = np.array([[days_since_high, pct_diff_from_high, days_since_low, pct_diff_from_low]])
    input_data = (input_data - scaler_mean) / scaler_scale

    # Predicting outcomes
    predictions = model.predict(input_data)
    pct_diff_high_next, pct_diff_low_next = predictions[0]
    return pct_diff_high_next, pct_diff_low_next

# Example usage (for testing purposes):
if __name__ == "__main__":
    # Train the model and print the accuracy
    train_model("output.csv")

    # Example prediction with sample values for the input features
    days_since_high = 1
    pct_diff_from_high = -2.62991414369633
    days_since_low = 1
    pct_diff_from_low = -1.60675501672902

    pct_diff_high_next, pct_diff_low_next = predict_outcomes(
        days_since_high, pct_diff_from_high, days_since_low, pct_diff_from_low
    )
    # Displaying the predicted values
    print(f"Predicted %_Diff_From_High_Next_1_Days: {pct_diff_high_next}")
    print(f"Predicted %_Diff_From_Low_Next_1_Days: {pct_diff_low_next}")