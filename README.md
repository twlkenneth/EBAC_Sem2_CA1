# Q2 Fraud Detection
#### Data Preparation:

KNN missing value imputer, Label Encoder, Feature Engineering, Standard Scaler, SMOTE

#### Binary classification using the following models:

Logistic Regression, Decision Tree, Naive Bayes Classifier, Random Forest, XGBoost, LightGBM, CatBoost, Tensorflow Multi Layer Peceptron, Encoder Decoder w KNN, Support Vector Machine

## Evaluation (normal)
```python
from q2_fraud_detection.model import LRegression

evaluation_metics = LRegression().run()

```
#### Output
```
{'auc_train': 0.6363008375787929, 'auc_valid': 0.6155861712594572, 
'acc_train': 0.6363008375787929, 'acc_valid': 0.7210041309183349, 
'matthew_corr_train': 0.2795725460502018, 'matthew_corr_valid': 0.14368101202678438, 
'f1_score_train': 0.5909090909090909, 'f1_score_valid': 0.22847100175746926}
```

## Evaluation (with extra parameters)
```python
from q2_fraud_detection.model import LRegression

evaluation_metics = LRegression(onehot_encode=True, polyfeature=True).run("evaluate", gridsearch=True)
```
#### Output
```
{'auc_train': 0.9999568258354201, 'auc_valid': 0.9033798049445029, 
'acc_train': 0.9999568258354201, 'acc_valid': 0.9574197648554179, 
'matthew_corr_train': 0.9999136553985353, 'matthew_corr_valid': 0.7488045651603159, 
'f1_score_train': 0.999956823971331, 'f1_score_valid': 0.7689655172413793}
```

## Predict (normal)
```python
from q2_fraud_detection.model import LRegression

prediction = LRegression().run("predict")
```
#### Output
```
        Insp
0          1
1          0
2          1
3          0
4          0
```

## Predict (with extra parameters)
```python
from q2_fraud_detection.model import LRegression

prediction = LRegression(onehot_encode=True, polyfeature=True).run("predict")
```
#### Output
```
        Insp
0          1
1          0
2          1
3          0
4          0
```

# Q3 Time Series Prediction

*Univariate*: StepWiseArima, Univariate Multi Step LSTM

*Multivariate*: Vector AutoRegression, Multivariate Multi Step LSTM

## Evaluation (Multivariate)
```python
from q3_time_series.model import VectorAutoRegression

evaluate_metrics = VectorAutoRegression().run("evaluate")
```
#### Output
```
[{'China': {'rmse_val': 0.10033340062857614, 'mae_val': 0.08651686327283652, 'mape_val': '0.08505443829568742 %'}}, 
{'India': {'rmse_val': 0.22945699053781246, 'mae_val': 0.16033218959840193, 'mape_val': '0.23784733993852525 %'}}, 
{'Singapore': {'rmse_val': 1.251872932175908, 'mae_val': 1.1537627135242137, 'mape_val': '1.4047582232972406 %'}}]
```

## Evaluation (Univariate)
```python


from q3_time_series.model import UnivariateMultiStepLSTM

evaluate_metrics = UnivariateMultiStepLSTM(3,2).run('Singapore', "evaluate")
```
#### Output
```
{'Singapore': {'rmse_train': 1.4304630397766138, 'rmse_val': 1.5401814488031627, 
                'mae_train': 1.2334213245388463, 'mae_val': 1.535661061311849, 
                'mape_train': '1.5851514521362737 %', 'mape_val': '1.8713769392807569 %'}}
```

## Predict (Multivariate)
```python
from q3_time_series.model import VectorAutoRegression

prediction = VectorAutoRegression().run("predict")
```
#### Output
```
          China      India  Singapore
2008  93.111763  70.077187  81.695653
2009  93.087358  70.002008  81.873667
```

## Predict (Univariate)
```python
from q3_time_series.model import UnivariateMultiStepLSTM

prediction = UnivariateMultiStepLSTM(3,2).run('Singapore', 'predict')
```
#### Output
```
      Singapore
2008  85.778084
2009  87.609962
```