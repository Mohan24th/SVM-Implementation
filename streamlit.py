# Import libraries
import streamlit as st
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# Title
st.title("SVR Regression App")

# Create dataset
x, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# Create model
svm_regressor = SVR(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    epsilon=0.1
)

# Train model
svm_regressor.fit(x_train, y_train)

# Predict test data
y_pred = svm_regressor.predict(x_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)

# Display MSE
st.subheader("Model Evaluation")
st.write("Mean Squared Error:", mse)

# User Input
st.subheader("Predict New Value")

input_value = st.number_input(
    "Enter feature value:",
    value=0.0
)

# Prediction button
if st.button("Predict"):

    # Convert input into 2D array
    new_data = np.array([[input_value]])

    # Predict
    prediction = svm_regressor.predict(new_data)

    # Output
    st.success(f"Predicted Value: {prediction[0]:.2f}")