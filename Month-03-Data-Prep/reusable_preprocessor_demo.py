"""
Month 03 — Project 2: Reusable Data Preprocessing Package Demo
Proves DataPreprocessor works on any tabular dataset — Titanic + Iris.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import load_iris

from data_preprocessor.preprocessor import DataPreprocessor, quick_clean

OUTPUT_DIR = "pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_titanic():
    print("=" * 60)
    print("DEMO 1: Titanic Dataset")
    print("=" * 60)

    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    df_clean = quick_clean(df)
    print("Quick cleaning complete")

    preprocessor = DataPreprocessor()
    preprocessor.fit_preprocessor(df_clean, target_col="Survived")

    X_transformed = preprocessor.transform(df_clean)
    print(f"Transformed shape: {X_transformed.shape}")

    pkl_path = f"{OUTPUT_DIR}/titanic_preprocessor.pkl"
    preprocessor.save(pkl_path)

    # Verify save/load round-trip
    loaded = DataPreprocessor.load(pkl_path)
    assert np.allclose(loaded.transform(df_clean), X_transformed), "Round-trip mismatch!"
    print("Save/load round-trip verified.")


def demo_iris():
    print("\n" + "=" * 60)
    print("DEMO 2: Iris Dataset (proves reusability — zero code changes)")
    print("=" * 60)

    raw = load_iris(as_frame=True)
    df = raw.frame.rename(columns={"target": "species"})

    # Inject missing values to exercise quick_clean
    rng = np.random.default_rng(42)
    df.loc[rng.choice(df.index, size=15, replace=False), "sepal length (cm)"] = np.nan

    df_clean = quick_clean(df)
    print(f"Loaded + cleaned Iris: {df_clean.shape[0]} rows")

    preprocessor = DataPreprocessor()
    preprocessor.fit_preprocessor(df_clean, target_col="species")

    X_transformed = preprocessor.transform(df_clean)
    print(f"Transformed shape: {X_transformed.shape}")
    preprocessor.save(f"{OUTPUT_DIR}/iris_preprocessor.pkl")


def main():
    demo_titanic()
    demo_iris()

    print(f"\nMonth 03 Project 2 Complete! — {datetime.now().strftime('%B %d, %Y')}")
    print("Key insight: same DataPreprocessor handles any dataset —")
    print("auto-detects types, imputes, scales, encodes, saves/loads.")


if __name__ == "__main__":
    main()
