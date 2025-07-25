# Kaggle-competitions
Repository where I present my work related to Kaggle competitions


## Predict the Introverts from the Extroverts
Link to competition:
[Predict the Introverts from the Extroverts](https://www.kaggle.com/competitions/playground-series-s5e7/overview)

Type of problem: Binary Classification

Evaluation Metric: Accuracy

The best model achieves a score of **0.977327** on the evaluation metric.

Summary of models
| Model Description    | Used frameworks | Private Score | Public Score |
|----------------------|-----------------|---------------|--------------|
| Base (all Extraverts)| None |  | 0.761133 |
| Logistic basic | sklearn |  | 0.974089 |
| Logistic with interactions | statsmodels |  | 0.974089 |
| xgboost | xgboost |  | <b>0.975708</b> |
| xgboost with optuna | xgboost |  | <b>0.975708</b> |
