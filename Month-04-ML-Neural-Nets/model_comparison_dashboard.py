import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime

plt.style.use('seaborn-v0_8')

def load_and_preprocess():
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)

    df = df.copy()
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']

    return X, y

def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    print(f"\n📊 {name}")
    print(f"   MAE: ${mae:,.2f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   R²: {r2:.4f}")
    print(f"   CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    return {
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'cv_r2': cv_scores.mean()
    }

def create_dashboard(results):
    df_results = pd.DataFrame(results)

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    sns.barplot(data=df_results, x='name', y='r2')
    plt.title('R² Score Comparison (Higher is Better)')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    sns.barplot(data=df_results, x='name', y='rmse')
    plt.title('RMSE Comparison (Lower is Better)')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    sns.barplot(data=df_results, x='name', y='mae')
    plt.title('MAE Comparison (Lower is Better)')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    sns.barplot(data=df_results, x='name', y='cv_r2')
    plt.title('Cross-Validation R² (Higher is Better)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Dashboard saved as 'model_comparison_dashboard.png'")
    plt.show()

def main():
    print("📊 Model Comparison Dashboard - House Price Prediction")
    print("="*70)

    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Support Vector Regressor": SVR()
    }

    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)

    create_dashboard(results)

    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, 'best_house_price_model.pkl')
    print("✅ Best model saved as best_house_price_model.pkl")

    print(f"\n🎉 Month 4 Project 2 Complete! - {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
