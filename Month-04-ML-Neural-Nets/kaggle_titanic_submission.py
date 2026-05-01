import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from datetime import datetime

TRAIN_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_data():
    print("📥 Downloading train data...")
    train = pd.read_csv(TRAIN_URL)
    test = pd.read_csv("test.csv")
    print(f"✅ Train: {train.shape} | Test: {test.shape}")
    return train, test

def preprocess(df, is_train=True):
    df = df.copy()

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Encode categoricals
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone',
                'Embarked_Q', 'Embarked_S']

    # Ensure all expected columns exist (test may be missing some dummies)
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

def main():
    print("🚢 Kaggle Titanic Submission Pipeline")
    print("="*70)

    train_raw, test_raw = load_data()

    X_train = preprocess(train_raw, is_train=True)
    y_train = train_raw['Survived']
    X_test = preprocess(test_raw, is_train=False)

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n📊 CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': test_raw['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"✅ submission.csv saved — {len(submission)} predictions")
    print("\n📤 Upload submission.csv at:")
    print("   https://www.kaggle.com/competitions/titanic/submit")
    print(f"\n🎉 Month 4 Project 3 Complete! - {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
