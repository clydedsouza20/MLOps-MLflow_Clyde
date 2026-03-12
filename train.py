import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model():
    # 1. Load the dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Set Experiment Name
    mlflow.set_experiment("California_Housing_Project")

    with mlflow.start_run():
        # 4. Define and train the model
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # 5. Make predictions
        y_pred = rf.predict(X_test)

        # 6. Calculate Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 7. Print to Terminal (So you can see them immediately!)
        print("-" * 30)
        print(" Run Finished!")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print("-" * 30)

        # 8. Log Parameters and Metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # 9. Log the Model
        mlflow.sklearn.log_model(rf, "random-forest-model")
        print("Model and metrics logged to MLflow successfully.")

if __name__ == "__main__":
    train_model()
