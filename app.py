import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained neural network model
saved = joblib.load("models/iris_model.joblib")

model = saved["model"]
target_names = saved["target_names"]

# Create the FastAPI application
app = FastAPI(title="Iris Neural Network API")


# Input data format
class FlowerData(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# Home page
@app.get("/")
def home():
    return {
        "message": "Iris Neural Network API is running"
    }


# Predict the flower species
@app.post("/predict")
def predict(data: FlowerData):

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "species": target_names[prediction]
    }