# Project Description

Fusion is a project for predicting targeted cash back. Objective is to build an end-to-end ML infrastructure for 
predicting targeted cash back. Given a consumer and a product,we want to know if we give it a special offer whether how
likely the consumer will buy the product.

More details : https://rakutenintelligence.atlassian.net/wiki/spaces/MLFUS

# How to Install

```sh
pip install git+ssh://git@github.rakops.com/americas-data/ml-fusion-tcb.git
```
or 

```sh
git clone git@github.rakops.com:americas-data/ml-fusion-tcb.git
cd ml-fusion-tcb
python setup.py install
```

## Using the Evaluation module

Evaluation Framework(fusion_tcb/metrics) in fusion_tcb module generates metrics to check performance of the models generated

### Regression Model Evaluation

```python
from fusion_tcb.metrics import regression

# Basket Prediction
y_pred = [9,10,15,8]
y = [10,12,15,11]
re = regression.RegressionEval(y_pred=y_pred, y_true=y, num_features=10)
eval_results_bp = regression.metrics(re)

print(eval_results_bp)

#output
RegressionMetrics(mean_squared_error=3.5, mean_absolute_error=1.5, r_squared=0.0, r_square_adjusted=-0.5)
```

### Classification Model Evaluation

```python
from fusion_tcb.metrics import classification

# Conversion Prediction
y_pred = [0.9,0.8,0.89,0.78,0.2,0.1,0.2,0.09,0.99,0.1]
y = [1,1,1,0,1,1,1,0,1,1]
ce = classification.ClassificationEval(y_pred=y_pred, y_true=y)
eval_results_ce = classification.metrics(ce)

#output
print(eval_results_ce.__dict__)
{   'accuracy': 0.5,
    'auc_prc': 0.9365079365079365,
    'auc_roc': 0.75,
    'f1score': 0.6153846153846154,
    'fvr': Fvr(fraction=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], recall_fvr=[0.0, 0.125, 0.25, 0.375, 0.5, 0.5, 0.625, 0.75, 0.875, 1.0], thr_fvr=[0.09, 0.1, 0.1, 0.2, 0.2, 0.78, 0.8, 0.89, 0.9, 0.99]),
    'prc': Prc(precision_thr=[0.8888888888888888, 0.8571428571428571, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0], recall_thr=[1.0, 0.75, 0.5, 0.5, 0.375, 0.25, 0.125, 0.0], thr_prc=[0.1, 0.2, 0.78, 0.8, 0.89, 0.9, 0.99]),
    'precision': 0.8,
    'recall': 0.5,
    'roc': Roc(fpr=[0.0, 0.0, 0.0, 0.5, 0.5, 1.0], tpr=[0.0, 0.125, 0.5, 0.5, 1.0, 1.0], thr_roc=[1.99, 0.99, 0.8, 0.78, 0.1, 0.09])}
```
