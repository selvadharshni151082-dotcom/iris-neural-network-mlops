import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)
print("Flower features:", iris.feature_names)
print("Flower classes:", iris.target_names)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 3: Create the Neural Network pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("neural_network", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    ))
])

# Step 4: Configure MLflow experiment tracking

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Neural_Network")


# Step 5: Train and track the model
with mlflow.start_run():
    model.fit(X_train, y_train)

    # Step 6: Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel training completed!")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    ))

    # Step 7: Log parameters and metrics in MLflow
    mlflow.log_param("model_type", "MLPClassifier")
    mlflow.log_param("hidden_layer_sizes", "(16, 8)")
    mlflow.log_param("activation", "relu")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", accuracy)

    # Step 8: Save the model and class names
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "target_names": iris.target_names.tolist()
        },
        "models/iris_model.joblib"
    )

    # Step 9: Log the trained model in MLflow
    
    mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    skops_trusted_types=[
        "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
    ]
    )


    print("\nModel saved successfully!")
    print("Model path: models/iris_model.joblib")
    print("MLflow tracking completed!")
