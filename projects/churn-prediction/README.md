# Telecom Customer Churn Prediction

End-to-end supervised-learning project: predict which telecom customers are at
high risk of churning, and surface the main drivers, so a retention team can act
before the customer leaves.

- **Source code:** [github.com/PersianSpock/Python-AI](https://github.com/PersianSpock/Python-AI)
- **Full notebook (with all outputs):** [Google Colab](https://colab.research.google.com/drive/12twa2VeJP1IL24zI3rVxQsB2hHPf9Kxo?usp=sharing)

> All metrics quoted below are the results reported in the
> [Colab notebook](https://colab.research.google.com/drive/12twa2VeJP1IL24zI3rVxQsB2hHPf9Kxo?usp=sharing).

## Problem

Telecom operators see an average **15–25% annual churn rate**, and acquiring a
new customer costs roughly **5–10× more** than retaining an existing one. That
makes early identification of at-risk customers a direct lever on profit. The
goal here is two-fold: build a model that flags high-risk customers during the
"action" phase (while intervention is still possible), and keep the model
interpretable enough to name the churn drivers.

## Dataset

The IBM **Telco Customer Churn** dataset — ~7,000 customers and 21 fields:
demographics (gender, senior citizen, partner, dependents), account details
(tenure, contract type, payment method, paperless billing, monthly and total
charges), subscribed services (phone, internet, online security/backup, device
protection, tech support, streaming), and the binary `Churn` target.

## Approach

**Feature engineering & cleaning.** `TotalCharges` coerced to numeric (with the
resulting nulls dropped), "No internet service" collapsed to "No" across the
service columns, categorical fields label-encoded and one-hot dummied, and all
features MinMax-scaled to [0, 1].

**Class imbalance.** The target is heavily imbalanced, so the minority (churn)
class is oversampled with **SMOTE** before the train/test split (70/30). PCA was
tried and rejected — it lowered ROC-AUC and destroyed the business
interpretability of the features, which was a stated priority.

**EDA & feature selection.** Correlation heatmaps, **chi-squared** scoring for
categorical features, **ANOVA F-test** for numerical features, and a **KMeans**
elbow analysis to probe latent structure.

**Classical models.** A bench of eight classifiers — XGBoost, Logistic
Regression, LightGBM, Random Forest, Decision Tree, Gradient Boosting, AdaBoost,
CatBoost — each scored with repeated stratified cross-validation and ROC-AUC,
then combined in a **stacking ensemble** (CatBoost as the final estimator).

**Neural approaches.** An **AutoKeras** `StructuredDataClassifier` search, plus a
hand-built **Keras ANN** (19 → 8 → 4 → 1, relu / relu / sigmoid, binary
cross-entropy). Four optimizers were compared (mini-batch SGD, momentum SGD,
RMSProp, Adam) against both up-sampled and down-sampled training sets.

**Tuning.** `GridSearchCV` on the two strongest models (CatBoost and XGBoost).

## Results

_As reported in the [Colab notebook](https://colab.research.google.com/drive/12twa2VeJP1IL24zI3rVxQsB2hHPf9Kxo?usp=sharing)._

| Model                         | Score |
| ----------------------------- | ----- |
| CatBoost (best classical)     | ~85%, **86% after GridSearch tuning** |
| XGBoost                       | 84.5% |
| Keras ANN (best: up-sampling, mini-batch SGD, lr=0.1) | 86% |

The tuned CatBoost and the up-sampled ANN landed at the same ~86% — i.e. once the
imbalance was handled and the gradient-boosted model was tuned, a compact 3-layer
network matched the best tree ensemble rather than beating it, which is a useful
finding in itself for a tabular problem of this size.

## How to run

The project was developed in Colab and reads the dataset from a mounted Drive
path. To run it locally:

```bash
git clone https://github.com/PersianSpock/Python-AI
cd Python-AI
pip install -r requirments.txt
# place WA_Fn-UseC_-Telco-Customer-Churn.csv and update the read_csv path in Churn.py
python Churn.py
```

Or open the [Colab notebook](https://colab.research.google.com/drive/12twa2VeJP1IL24zI3rVxQsB2hHPf9Kxo?usp=sharing)
and run all cells — it already contains the full outputs and plots.

## Saved artifacts

The repo ships the trained network: `model.yaml` holds the architecture
(serialized via Keras `to_json()`, despite the `.yaml` extension) and
`fina_churn_model.h5` holds the saved weights.
