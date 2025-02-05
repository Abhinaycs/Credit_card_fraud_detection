import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the saved ensemble model (VotingClassifier)
with open("voting_classifier_model.pkl", 'rb') as f:
    ensemble_model = pickle.load(f)

# Load the scaler (MinMaxScaler)
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Streamlit title
st.title("Credit Card Fraud Detection")

# Streamlit sidebar for user input
st.sidebar.header("Enter Transaction Details")

# Take the features as a single input string, separated by commas
input_str = st.sidebar.text_area("Enter 30 feature values (comma-separated)", 
                                value="", 
                                height=150)

# If the user enters some data, process it
if input_str:
    try:
        # Convert the input string to a list of floats
        input_list = [float(i) for i in input_str.split(",")]

        if len(input_list) != 30:
            st.error("Please enter exactly 30 feature values.")
        else:
            # Reshape the input data to match the model's input
            input_data = np.array(input_list).reshape(1, -1)

            # Scale the input data using the scaler
            input_data_scaled = scaler.transform(input_data)

            # Make prediction using the ensemble model (VotingClassifier)
            ensemble_prediction = ensemble_model.predict(input_data_scaled)
            ensemble_probability = ensemble_model.predict_proba(input_data_scaled)[:, 1][0]

            # Display the results
            st.subheader("Prediction Results")
            st.write(f"**Ensemble Model Prediction**: {'Fraudulent' if ensemble_prediction[0] == 1 else 'Non-Fraudulent'}")
            st.write(f"**Ensemble Model Fraud Probability**: {ensemble_probability:.4f}")
    
    except ValueError:
        st.error("Invalid input. Please enter numerical values separated by commas.")
