# Machine Learning Assignment 2

## Problem Statement
The objective of this project is to implement multiple classification models
on a chosen dataset and evaluate their performance using standard metrics.

## Dataset Description
The Heart Disease dataset is used for this project. The dataset contains
clinical parameters used to predict the presence of heart disease.

- Instances: ~1000
- Features: 13
- Target Variable: target (0 = No Disease, 1 = Disease)

## Models Used

+----------------------+----------+----------+-----------+----------+----------+----------+
| Model                | Accuracy |   AUC    | Precision |  Recall  |    F1    |   MCC    |
+----------------------+----------+----------+-----------+----------+----------+----------+
| Logistic Regression  | 0.880435 | 0.935728 | 0.833333  | 0.652174 | 0.731707 | 0.664411 |
| Decision Tree        | 0.858696 | 0.789855 | 0.750000  | 0.652174 | 0.697674 | 0.608581 |
| KNN                  | 0.885870 | 0.893589 | 0.837838  | 0.673913 | 0.746988 | 0.681082 |
| Naive Bayes          | 0.804348 | 0.900599 | 0.567568  | 0.913043 | 0.700000 | 0.601527 |
| Random Forest        | 0.885870 | 0.916588 | 0.857143  | 0.652174 | 0.740741 | 0.679565 |
| XGBoost              | 0.869565 | 0.892250 | 0.805556  | 0.630435 | 0.707317 | 0.632772 |
+----------------------+----------+----------+-----------+----------+----------+----------+


## Observations
- Logistic Regression: Provides strong overall performance with high AUC, indicating effective class separation. High precision but comparatively lower recall suggests conservative prediction behaviour. Serves as a reliable baseline model.

- Decision Tree: Achieves good accuracy while capturing nonlinear relationships. Moderate AUC indicates reasonable discrimination ability. Performance may be affected by slight overfitting compared to ensemble models.

- kNN: Delivers one of the highest accuracy scores, demonstrating effective classification with scaled features. Balanced precision and recall indicate stable prediction behaviour, though performance is sensitive to feature scaling.

- Naive Bayes: Exhibits very high recall, showing strong ability to detect positive cases. Lower precision indicates increased false positives. Performs reasonably well despite independence assumptions.

- Random Forest (Ensemble): Produces high accuracy and strong AUC due to ensemble learning. Demonstrates improved robustness and reduced overfitting compared to individual tree-based models.

- XGBoost (Ensemble): Achieves competitive accuracy with strong AUC. Gradient boosting enhances predictive performance by sequentially correcting errors, resulting in balanced classification behaviour.


