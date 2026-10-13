
import joblib
import numpy as np
import logging
import time

from collections import deque
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Configure logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Load the trained neural network model
# --------------------------------------------------

saved = joblib.load("models/iris_model.joblib")

model = saved["model"]
target_names = saved["target_names"]


# --------------------------------------------------
# 3. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(title="Iris Neural Network API")


# --------------------------------------------------
# 4. Prepare reference data for drift monitoring
# --------------------------------------------------

reference_features = load_iris().data

feature_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

# Keep only the latest 500 prediction inputs
app.state.recent_inputs = deque(maxlen=500)


# --------------------------------------------------
# 5. Initialize API monitoring counters
# --------------------------------------------------

app.state.total_requests = 0
app.state.successful_requests = 0
app.state.failed_requests = 0


# --------------------------------------------------
# 6. Monitor prediction requests and response time
# --------------------------------------------------

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

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
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

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


# --------------------------------------------------
# 7. Define and validate prediction input
# --------------------------------------------------

class FlowerData(BaseModel):

    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# --------------------------------------------------
# 8. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Iris Neural Network API is running"
    }


# --------------------------------------------------
# 9. Predict flower species and record input data
# --------------------------------------------------

@app.post("/predict")
def predict(data: FlowerData):

    input_features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    # Store inputs for future drift analysis
    app.state.recent_inputs.append(input_features)

    features = np.array([input_features])

    prediction = int(model.predict(features)[0])

    species = target_names[prediction]

    return {
        "prediction": prediction,
        "species": species
    }


# --------------------------------------------------
# 10. Data drift monitoring endpoint
# --------------------------------------------------

@app.get("/monitoring/drift")
def monitor_drift():

    current_inputs = np.array(
        list(app.state.recent_inputs),
        dtype=float
    )

    sample_count = len(current_inputs)

    # Require at least 30 incoming samples
    if sample_count < 30:

        return {
            "status": "insufficient_data",
            "samples_collected": sample_count,
            "required_samples": 30,
            "message": (
                "Make at least 30 valid prediction requests "
                "before checking data drift."
            )
        }

    feature_results = {}

    # Compare each incoming feature with the reference data
    for index, feature_name in enumerate(feature_names):

        reference_values = reference_features[:, index]
        incoming_values = current_inputs[:, index]

        test_result = ks_2samp(
            reference_values,
            incoming_values
        )

        p_value = float(test_result.pvalue)

        feature_results[feature_name] = {
            "p_value": round(p_value, 4),
            "drift_detected": bool(p_value < 0.05)
        }

    # Check whether any feature shows a possible shift
    drift_detected = any(
        result["drift_detected"]
        for result in feature_results.values()
    )

    if drift_detected:
        status = "potential_drift_detected"
    else:
        status = "no_significant_drift_detected"

    logger.info(
        "Data drift check: status=%s samples=%s",
        status,
        sample_count
    )

    return {
        "status": status,
        "samples_collected": sample_count,
        "reference_samples": len(reference_features),
        "significance_threshold": 0.05,
        "features": feature_results
    }
