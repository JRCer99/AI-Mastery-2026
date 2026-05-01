# Month 04: Machine Learning & Neural Networks

**Focus:** Build, evaluate, and compare ML models end-to-end. Understand train/eval loops, overfitting, and model selection.

---

## Learning Goals

- Build full ML pipelines: data → features → train → evaluate → deploy-ready
- Understand bias-variance tradeoff, overfitting, regularization
- Compare multiple algorithms with cross-validation
- Submit to a real Kaggle competition
- Understand how a basic neural network learns (backprop, activations)

---

## Projects

- [ ] Project 1: End-to-End ML Pipeline (House Price Prediction)
  - Dataset: California Housing (sklearn) or Kaggle Housing Prices
  - Models: Linear Regression, Ridge, Random Forest, Gradient Boosting
  - Output: trained model, evaluation report, feature importance plot
  - Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

- [ ] Project 2: Model Comparison Dashboard (Scikit-learn)
  - Compare 5+ classifiers on a benchmark dataset
  - Metrics: accuracy, precision, recall, F1, ROC-AUC
  - Output: visual comparison dashboard (matplotlib/seaborn)

- [ ] Project 3: First Kaggle Competition Submission
  - Pick active beginner comp (Titanic or Housing Prices)
  - Submit predictions, document leaderboard score
  - Iterate at least once based on score feedback

---

## Key Concepts

| Concept | Why It Matters |
|---|---|
| Train/val/test split | Honest evaluation, no data leakage |
| Cross-validation (k-fold) | Robust performance estimate |
| Overfitting / regularization | Generalization to unseen data |
| Hyperparameter tuning (GridSearch) | Squeeze out performance |
| Feature importance | Interpretability + debugging |
| Confusion matrix / ROC curve | Beyond accuracy for imbalanced data |

---

## Folder Structure

```
Month-04-ML-Neural-Nets/
├── house_price_pipeline.py     # Project 1
├── model_comparison.py         # Project 2
├── kaggle_submission.py        # Project 3
├── notebooks/                  # exploratory work
└── data/                       # datasets (gitignored if large)
```
