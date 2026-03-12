# MLOps Lab: California Housing Price Prediction with MLflow

This project demonstrates a complete machine learning workflow using **MLflow** for experiment tracking. It uses the California Housing dataset to train a Random Forest Regressor and logs parameters, metrics, and models.

## 🚀 Project Overview
The goal of this lab was to implement a reproducible ML pipeline that tracks model performance across different runs.

### Key Features:
* **Dataset:** Scikit-learn's California Housing dataset.
* **Model:** Random Forest Regressor.
* **Tracking:** MLflow is used to log $R^2$, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
* **Versioning:** Automated model logging and artifact storage.

---

## 📊 Model Performance
After tuning the hyperparameters (e.g., `n_estimators=100`, `max_depth=10`), the model achieved the following results:

| Metric | Value |
| :--- | :--- |
| **R² Score** | 0.7756 |
| **Mean Squared Error (MSE)** | 0.2554 |
| **Mean Absolute Error (MAE)** | 0.3297 |
---

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/clydedsouza20/MLOps-MLflow_Clyde.git](https://github.com/clydedsouza20/MLOps-MLflow_Clyde.git)
   cd MLOps-MLflow_Clyde
