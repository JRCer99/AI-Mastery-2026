import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)

    features = ['Pclass', 'Sex', 'AgeBin', 'FareBin', 'FamilySize', 'IsAlone',
                'Title', 'Embarked_Q', 'Embarked_S']

    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

def main():
    print("🚢 Kaggle Titanic Submission Pipeline (v2 — improved features)")
    print("="*70)

    train_raw, test_raw = load_data()

    X_train = preprocess(train_raw)
    y_train = train_raw['Survived']
    X_test = preprocess(test_raw)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_split=4, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    }

    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"\n📊 {name}")
        print(f"   CV Accuracy: {cv.mean():.4f} (±{cv.std():.4f})")
        if cv.mean() > best_score:
            best_score = cv.mean()
            best_model = model
            best_name = name

    print(f"\n🏆 Best: {best_name} ({best_score:.4f})")
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)
    submission = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': predictions})
    submission.to_csv('submission.csv', index=False)
    print(f"✅ submission.csv saved — {len(submission)} predictions")
    print("\n📤 Upload at: https://www.kaggle.com/competitions/titanic/submit")
    print(f"\n🎉 {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
