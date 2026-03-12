import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load unique dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name
mlflow.set_experiment("Clyde_Housing_MLOps")

with mlflow.start_run(run_name="Initial_Model"):
    # Log parameters
    params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    mlflow.log_params(params)

    # Train
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    # Log metrics
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mlflow.log_metric("r2_score", r2)
    
    # Log model artifact
    mlflow.sklearn.log_model(model, "model")

    print(f"Run Finished! R2 Score: {r2:.4f}")