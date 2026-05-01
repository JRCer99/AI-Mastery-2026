# EDA Report — Titanic Dataset
*Generated: April 30, 2026 at 21:43*

## Dataset Overview
- **Rows:** 891
- **Columns:** 12
- **Missing values:** 866 total
- **Numeric columns:** 7
- **Survival rate:** 38.4%

## Summary Statistics
|       |   PassengerId |   Survived |   Pclass |     Age |   SibSp |   Parch |    Fare |
|:------|--------------:|-----------:|---------:|--------:|--------:|--------:|--------:|
| count |       891     |    891     |  891     | 714     | 891     | 891     | 891     |
| mean  |       446     |      0.384 |    2.309 |  29.699 |   0.523 |   0.382 |  32.204 |
| std   |       257.354 |      0.487 |    0.836 |  14.526 |   1.103 |   0.806 |  49.693 |
| min   |         1     |      0     |    1     |   0.42  |   0     |   0     |   0     |
| 25%   |       223.5   |      0     |    2     |  20.125 |   0     |   0     |   7.91  |
| 50%   |       446     |      0     |    3     |  28     |   0     |   0     |  14.454 |
| 75%   |       668.5   |      1     |    3     |  38     |   1     |   0     |  31     |
| max   |       891     |      1     |    3     |  80     |   8     |   6     | 512.329 |

## Missing Values
- **Age**: 177 (19.9%)
- **Cabin**: 687 (77.1%)
- **Embarked**: 2 (0.2%)

## Visualizations

### Distributions
![Distributions](eda_output/distributions.png)

### Correlation Heatmap
![Correlation Heatmap](eda_output/correlation_heatmap.png)

### Missing Values
![Missing Values](eda_output/missing_values.png)

## Next Steps
- Handle missing values (imputation strategies)
- Feature engineering for ML pipeline (Month 04)
- Build predictive model (survival classification)

---
*AI Mastery 2026 — Month 02, Project 1*