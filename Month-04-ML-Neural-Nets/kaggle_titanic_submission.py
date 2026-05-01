import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from datetime import datetime

TRAIN_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_data():
    print("📥 Downloading train data...")
    train = pd.read_csv(TRAIN_URL)
    test = pd.read_csv("test.csv")
    print(f"✅ Train: {train.shape} | Test: {test.shape}")
    return train, test

def extract_title(df):
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
    )
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
    df['Title'] = df['Title'].fillna(0)
    return df

def preprocess(df):
    df = df.copy()
    df = extract_title(df)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone',
                'Title', 'Embarked_Q', 'Embarked_S']

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

def main():
    print("🚢 Kaggle Titanic Submission Pipeline (v3 — ensemble)")
    print("="*70)

    train_raw, test_raw = load_data()
    X_train = preprocess(train_raw)
    y_train = train_raw['Survived']
    X_test = preprocess(test_raw)

    rf = RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_split=4,
                                min_samples_leaf=2, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=42)

    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')

    for name, model in [("Random Forest", rf), ("Gradient Boosting", gb), ("Ensemble", ensemble)]:
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"\n📊 {name}: {cv.mean():.4f} (±{cv.std():.4f})")

    print("\n🏆 Training ensemble on full data...")
    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)
    submission = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': predictions})
    submission.to_csv('submission.csv', index=False)
    print(f"✅ submission.csv saved — {len(submission)} predictions")
    print("\n📤 Upload at: https://www.kaggle.com/competitions/titanic/submit")
    print(f"\n🎉 {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
