import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

def main():
    print("🚀 MLOps Project: MLflow + House Price Model")
    print("="*60)

    mlflow.set_experiment("House_Price_Prediction")

    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df = df.copy()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())  # fillna before ratio
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {"n_estimators": 150, "max_depth": None, "random_state": 42, "model": "RandomForest"}

    with mlflow.start_run(run_name="RandomForest_v1"):
        mlflow.log_params(params)
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²: {r2:.4f}")

    model_path = Path(__file__).parent / "house_price_model.pkl"
    joblib.dump(model, model_path)
    print("✅ Model saved with MLflow tracking!")

    print(f"\n🎉 Month 8 Project 1 Complete! - {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
