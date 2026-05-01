"""
Month 03 — Project 1: Titanic Full Data Pipeline
End-to-end: raw CSV → feature engineering → preprocessing → ML-ready split.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUTPUT_DIR = "pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    print("\n" + "=" * 60)
    print("INITIAL DATA EXPLORATION")
    print("=" * 60)
    print(df.head())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nData Types:\n", df.dtypes)


def feature_engineering(df):
    df = df.copy()

    # Fill missing values FIRST so derived features don't inherit NaN
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Title from name
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare, "Rare")
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace(["Mme"], "Mrs")

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age groups — string dtype so OneHotEncoder handles it cleanly
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    ).astype(str)

    # Drop high-cardinality / ID columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    return df


def visualize(df):
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.countplot(data=df, x="Title", hue="Survived", ax=axes[0, 0])
    axes[0, 0].set_title("Survival by Title")

    sns.countplot(data=df, x="FamilySize", hue="Survived", ax=axes[0, 1])
    axes[0, 1].set_title("Survival by Family Size")

    sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Age Distribution by Survival")

    sns.histplot(data=df, x="Fare", hue="Survived", bins=40, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Fare Distribution by Survival")

    plt.suptitle("Titanic — Engineered Features", fontsize=14)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/feature_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


def build_pipeline():
    numeric = ["Age", "Fare", "FamilySize", "IsAlone", "SibSp", "Parch"]
    categorical = ["Sex", "Pclass", "Embarked", "Title", "AgeGroup"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor)])


def main():
    df = load_data()
    explore_data(df)

    df = feature_engineering(df)
    print(f"\nAfter feature engineering: {df.shape[1]} columns")
    print(df.head())

    visualize(df)

    cleaned_path = f"{OUTPUT_DIR}/titanic_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataset saved → {cleaned_path}")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    pipeline = build_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train_processed)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test_processed)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train.values)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test.values)
    print(f"ML-ready arrays saved → {OUTPUT_DIR}/")

    pkl_path = f"{OUTPUT_DIR}/preprocessor.pkl"
    joblib.dump(pipeline, pkl_path)
    print(f"Preprocessing pipeline saved → {pkl_path}")

    print(f"\nMonth 03 Project 1 Complete! — {datetime.now().strftime('%B %d, %Y')}")
    print("Key insight: fill nulls BEFORE deriving features, or NaN propagates silently.")


if __name__ == "__main__":
    main()
