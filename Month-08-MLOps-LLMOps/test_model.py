import pytest
import joblib
import pandas as pd


def test_model_exists():
    model = joblib.load("house_price_model.pkl")
    assert model is not None


def test_model_prediction():
    model = joblib.load("house_price_model.pkl")
    # drop_first=True drops <1H OCEAN (alphabetically first), so 4 dummies remain
    sample = pd.DataFrame({
        'longitude': [-122.23], 'latitude': [37.88], 'housing_median_age': [41],
        'total_rooms': [880], 'total_bedrooms': [129], 'population': [322],
        'households': [126], 'median_income': [8.3252],
        'ocean_proximity_INLAND': [0],
        'ocean_proximity_ISLAND': [0],
        'ocean_proximity_NEAR BAY': [1],
        'ocean_proximity_NEAR OCEAN': [0],
        'rooms_per_household': [7.0],
        'bedrooms_ratio': [0.15],
        'population_per_household': [2.55]
    })
    prediction = model.predict(sample)
    assert prediction[0] > 0
