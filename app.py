import joblib
import numpy as np
import logging
import time

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

# Configure monitoring logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Load the trained neural network model
saved = joblib.load("models/iris_model.joblib")

model = saved["model"]
target_names = saved["target_names"]

# Create the FastAPI application
app = FastAPI(title="Iris Neural Network API")

# Monitoring counters
app.state.total_requests = 0
app.state.successful_requests = 0
app.state.failed_requests = 0


# Monitor prediction requests
@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    # Monitor only the prediction endpoint
    if request.url.path != "/predict":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500

    app.state.total_requests += 1

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        logger.exception("Prediction request encountered an error")
        raise

    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000

        if status_code < 400:
            app.state.successful_requests += 1
        else:
            app.state.failed_requests += 1

        logger.info(
            "endpoint=/predict status=%s duration_ms=%.2f "
            "total=%s successful=%s failed=%s",
            status_code,
            duration_ms,
            app.state.total_requests,
            app.state.successful_requests,
            app.state.failed_requests
        )


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

    species = target_names[prediction]

    return {
        "prediction": prediction,
        "species": species
    }
