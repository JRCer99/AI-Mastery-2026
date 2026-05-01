# Month 03: Exploratory Data Manipulation & Feature Engineering

**Focus:** Mastering Pandas, data cleaning, feature engineering, and building reusable ML-ready pipelines.

---

## 🎯 Learning Goals

- Handle missing data with appropriate imputation strategies
- Encode categorical variables (one-hot, label, ordinal)
- Engineer new features from raw data (binning, log transforms, interactions)
- Scale and normalize features for ML algorithms
- Build a reusable preprocessing pipeline class

---

## 📅 Projects

- [ ] Project 1: Titanic / Kaggle Full Data Pipeline
  - End-to-end: raw data → cleaned → feature-engineered → ML-ready CSV
  - Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

- [ ] Project 2: Reusable Automated Data Preprocessing Package
  - Class-based `DataPipeline` transformer reusable across any dataset
  - Libraries: `pandas`, `numpy`, `scikit-learn`

---

## 🧠 Key Concepts

| Concept | Why It Matters for AI |
|---|---|
| Missing value imputation | Dirty data breaks every ML model |
| One-hot / label encoding | ML models require numeric input |
| Feature scaling | Prevents large-magnitude features dominating |
| Feature engineering | Domain knowledge → better signal |
| Train/test split (stratified) | Prevents data leakage into evaluation |

---

## 📚 Resources

- [Kaggle — Pandas Course](https://www.kaggle.com/learn/pandas)
- [Scikit-learn — Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Towards Data Science — Feature Engineering Guide](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)

---

## 📂 Structure

```
Month-03-Data-Prep/
├── titanic_pipeline.py        # Project 1
├── data_pipeline.py           # Project 2
└── data/                      # datasets
```
