import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

# Load dataset (Ames Housing - classic regression dataset)
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    print(f"✅ Loaded housing dataset: {df.shape[0]} houses, {df.shape[1]} features")
    return df

def preprocess_data(df):
    df = df.copy()
    # Simple feature engineering
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    # Fill missing values
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

    # Convert ocean proximity to numeric (one-hot would be better, but keeping simple for now)
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    return df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Model Performance:")
    print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"Root Mean Squared Error: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

    return model, X_test, y_test, y_pred

def main():
    print("🏠 End-to-End ML Pipeline - House Price Prediction")
    print("="*70)

    df = load_data()
    df = preprocess_data(df)

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    model, X_test, y_test, y_pred = train_model(X, y)

    joblib.dump(model, 'house_price_model.pkl')
    print("✅ Model saved as house_price_model.pkl")

    print(f"\n🎉 Month 4 Project 1 Complete! - {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
