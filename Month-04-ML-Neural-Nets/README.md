# Month 04: Machine Learning & Neural Networks

**Focus:** Supervised learning, model evaluation, feature importance, and building end-to-end ML pipelines with Scikit-learn.

---

## 🎯 Learning Goals

- Build full ML pipelines: raw data → features → train → evaluate → save
- Understand bias-variance tradeoff, overfitting, and regularization
- Compare multiple algorithms using cross-validation and metrics
- Interpret model performance with MAE, RMSE, R², and feature importance
- Submit to a real Kaggle competition

---

## 📅 Projects

- [x] Project 1: End-to-End ML Pipeline (House Price Prediction) ✅
  - Dataset: California Housing (via sklearn/handson-ml2)
  - Model: Random Forest Regressor
  - Output: trained model (.pkl), MAE/RMSE/R² report
  - Libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`

- [ ] Project 2: Model Comparison Dashboard (Scikit-learn)
  - Compare 5+ classifiers on a benchmark dataset
  - Metrics: accuracy, precision, recall, F1, ROC-AUC
  - Output: visual comparison dashboard (matplotlib/seaborn)

- [ ] Project 3: First Kaggle Competition Submission
  - Pick active beginner comp (Titanic or Housing Prices)
  - Submit predictions, document leaderboard score
  - Iterate at least once based on score feedback

---

## 🧠 Key Concepts

| Concept | Why It Matters |
|---|---|
| Train/val/test split | Honest evaluation, no data leakage |
| Cross-validation (k-fold) | Robust performance estimate |
| MAE / RMSE / R² | Regression evaluation metrics |
| Feature importance | Interpretability + model debugging |
| Overfitting / regularization | Generalization to unseen data |
| Hyperparameter tuning | Squeeze out more performance |

---

## 📚 Resources

- [Scikit-learn — Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [Hands-On ML (Géron) — Chapter 2](https://github.com/ageron/handson-ml2)
- [Kaggle — Intro to ML Course](https://www.kaggle.com/learn/intro-to-machine-learning)

---

## 📂 Structure

```
Month-04-ML-Neural-Nets/
├── house_price_pipeline.py     # Project 1: End-to-End ML Pipeline
├── model_comparison.py         # Project 2: Model Comparison Dashboard
├── kaggle_submission.py        # Project 3: Kaggle Submission
├── house_price_model.pkl       # Saved model (gitignored if large)
└── data/                       # datasets (gitignored if large)
```
