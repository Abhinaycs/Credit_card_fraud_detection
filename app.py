from fastapi import FastAPI
import numpy as np
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the saved model and scaler
import pickle

# Load the saved model
with open("voting_classifier_model.pkl", 'rb') as f:
    model = pickle.load(f)

# model = joblib.load("voting_classifier_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input data model
class TransactionData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API"}

@app.post("/predict")
def predict(data: TransactionData):
    try:
        # Convert input to numpy array and reshape
        input_data = np.array(data.features).reshape(1, -1)
        
        # Scale input data
        input_data_scaled = scaler.transform(input_data)

        # Get prediction and probability
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)[:, 1][0]

        return {
            "prediction": int(prediction[0]),
            "fraud_probability": round(probability, 4)
        }
    except Exception as e:
        return {"error": str(e)}
