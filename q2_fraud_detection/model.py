import numpy as np
import pandas as pd
from typing import Dict, Union

import lightgbm as lgb
from catboost import CatBoostClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import xgboost as xgb

from q2_fraud_detection.base import Base

__all__ = ['LRegression', 'DecisionTree', 'NaiveBayesClassifier', 'RandomForest', 'XGBoost', 'TensorflowMLP',
           'LightGBM', 'EncoderDecoderKNN', 'CatBoost', 'SupportVectorMachine']

"""
class with Grid Search Function: LRegression, LightGBM, RandomForest, XGBoost, CatBoost, SupportVectorMachine
class requiring threshold input for classification: TensorflowMLP
"""


class LRegression(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.space = {'warm_start': hp.choice('warm_start', [True, False]),
                        'fit_intercept': hp.choice('fit_intercept', [True, False]),
                        'tol': hp.uniform('tol', 0.00001, 0.0001),
                        'C': hp.uniform('C', 1, 10),
                        'max_iter' : hp.choice('max_iter', range(100,1000)),
                        'multi_class' : 'auto',}
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode

    def run(self, action: str = 'evaluate', gridsearch = False) -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import LRegression
        >>>
        >>> # To Evaluate
        >>> evaluate_metrics = LRegression().run("evaluate")
        >>> # To Predict
        >>> prediction = LRegression().run("predict")
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)
        lr = LogisticRegression()

        if gridsearch == True:
            trials = Trials()
            best_hyperparams = fmin(fn=self.objective,
                                    space=self.space,
                                    algo=tpe.suggest,
                                    max_evals=100,
                                    trials=trials)
            lr_best = LogisticRegression(**best_hyperparams)
        else:
            lr_best = lr
        lr_best.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(lr)
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, lr_best.predict(X_valid)), title='LRegression')
            return self._evaluate(lr_best, X_train_res, X_valid, y_train_res, y_valid)

    def objective(self, space: Dict) -> Dict:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()

        lr = LogisticRegression(
            warm_start=space['warm_start'], fit_intercept=(space['fit_intercept']), tol=space['tol'],
            C=int(space['C']), max_iter=int(space['max_iter']), multi_class=(space['multi_class']))

        lr.fit(X_train_res, y_train_res)

        pred = lr.predict(X_valid)
        auc_score = roc_auc_score(y_valid, pred)

        return {'loss': auc_score, 'status': STATUS_OK}


class DecisionTree(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import DecisionTree
        >>>
        >>> # To Evaluate
        >>> evaluate_metrics = DecisionTree().run("evaluate")
        >>> # To Predict
        >>> prediction = DecisionTree().run("predict")
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(clf)
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, clf.predict(X_valid)), title='DecisionTree')
            return self._evaluate(clf, X_train_res, X_valid, y_train_res, y_valid)


class NaiveBayesClassifier(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import NaiveBayesClassifier
        >>>
        >>> # To Evaluate
        >>> evaluate_metrics = NaiveBayesClassifier().run("evaluate")
        >>> # To Predict
        >>> prediction = NaiveBayesClassifier().run("predict")
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)
        nbc = GaussianNB()
        nbc.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(nbc)
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, nbc.predict(X_valid)), title='NaiveBayesClassifier')
            return self._evaluate(nbc, X_train_res, X_valid, y_train_res, y_valid)


class RandomForest(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.param_grid =  {'n_estimators': [200, 500],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'max_depth' : [4,5,6,7,8],
                            'criterion' :['gini', 'entropy']}
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode

    def run(self, action: str = 'evaluate', gridsearch = False) -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import RandomForest
        >>>
        >>> # To Evaluate without gridsearch
        >>> evaluate_metrics = RandomForest().run("evaluate")
        >>> # To Predict with gridsearch
        >>> prediction = RandomForest().run("predict", True)
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)

        rf = RandomForestClassifier(random_state=0)
        if gridsearch == True:
            grid_rf = GridSearchCV(estimator=rf, cv=5, param_grid=self.param_grid, scoring='roc_auc')
            grid_rf.fit(X_train_res, y_train_res.ravel())
            rf_best = RandomForestClassifier(**grid_rf.best_params_)
        else:
            rf_best = rf

        rf_best.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(rf_best)
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, rf_best.predict(X_valid)),
                                       title=f'RandomForest_GSearch[{gridsearch}]')
            return self._evaluate(rf_best, X_train_res, X_valid, y_train_res, y_valid)


class XGBoost(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
                      'gamma': hp.uniform('gamma', 1, 9),
                      'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                      'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                      'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                      'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                      'n_estimators': 2000,
                      # 'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
                      'seed': 0}
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode

    def run(self, action: str = 'evaluate', gridsearch = False) -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import XGBoost
        >>>
        >>> # To Evaluate without gridsearch
        >>> evaluate_metrics = XGBoost().run("evaluate")
        >>> # To Predict with gridsearch
        >>> prediction = XGBoost().run("predict", True)
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)
        if gridsearch == True:
            trials = Trials()
            best_hyperparams = fmin(fn=self.objective,
                                    space=self.space,
                                    algo=tpe.suggest,
                                    max_evals=100,
                                    trials=trials)
            best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
            clf_xgb = xgb.XGBClassifier(**best_hyperparams)
        else:
            clf_xgb = xgb.XGBClassifier()

        clf_xgb.fit(X_train_res, y_train_res)

        if action == 'predict':
            return self._predict(clf_xgb)
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, clf_xgb.predict(X_valid)),
                                       title=f'XGBoost_GSearch[{gridsearch}]')
            return self._evaluate(clf_xgb, X_train_res, X_valid, y_train_res, y_valid)

    def objective(self, space: Dict) -> Dict:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()

        clf = xgb.XGBClassifier(
            n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
            reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
            colsample_bytree=int(space['colsample_bytree']))

        evaluation = [(X_train_res, y_train_res), (X_valid, y_valid.ravel())]

        clf.fit(X_train_res, y_train_res,
                eval_set=evaluation, eval_metric=self.evalmcc,
                early_stopping_rounds=10, verbose=False)

        pred = clf.predict(X_valid)
        auc_score = roc_auc_score(y_valid, pred)

        return {'loss': auc_score, 'status': STATUS_OK}

    @staticmethod
    def evalmcc(preds, dtrain):
        THRESHOLD = 0.5
        labels = dtrain.get_label()
        return 'MCC', matthews_corrcoef(labels, preds >= THRESHOLD)


class TensorflowMLP(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.METRICS = [
            # tf.keras.metrics.TruePositives(name='tp'),
            # tf.keras.metrics.FalsePositives(name='fp'),
            # tf.keras.metrics.TrueNegatives(name='tn'),
            # tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            # tf.keras.metrics.Precision(name='precision'),
            # tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import TensorflowMLP
        >>>
        >>> # To Evaluate
        >>> evaluate_metrics = TensorflowMLP().run("evaluate")
        >>> # To Predict
        >>> prediction = TensorflowMLP().run("predict")
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)
        model = self.make_model(X_train_res)

        baseline_history = model.fit(
            X_train_res,
            y_train_res,
            batch_size=128,
            epochs=50,
            callbacks=[self.early_stopping],
            validation_data=(X_valid, y_valid))

        if action == 'predict':
            return self._predict(model)
        else:
            y_pred = model.predict(X_valid)
            self.plot_confusion_matrix(confusion_matrix(y_valid, y_pred > 0.5), title='TensorflowMLP')
            return self._evaluate(model, X_train_res, X_valid, y_train_res, y_valid, 0.5)

    def make_model(self, X_train, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                128, activation='relu',
                input_shape=(X_train.shape[-1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=self.METRICS)

        return model


class LightGBM(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.class_weight = [None, 'balanced']
        self.boosting_type = ['gbdt', 'goss', 'dart']
        self.num_leaves = [30, 50, 100, 150]
        self.learning_rate = list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=10))
        self.lgg_grid = dict(class_weight=self.class_weight, boosting_type=self.boosting_type,
                             num_leaves=self.num_leaves,
                             learning_rate=self.learning_rate)
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode

    def run(self, action: str = 'evaluate', gridsearch = False) -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import LightGBM
        >>>
        >>> # To Evaluate without gridsearch
        >>> evaluate_metrics = LightGBM().run("evaluate")
        >>> # To Predict with gridsearch
        >>> prediction = LightGBM().run("predict", True)
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)

        lgg = lgb.LGBMClassifier()
        if gridsearch == True:
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                         random_state=1)
            grid_search = GridSearchCV(estimator=lgg,
                                       param_grid=self.lgg_grid, n_jobs=-1, cv=cv,
                                       scoring='roc_auc', error_score=0)
            grid_search.fit(X_train_res, y_train_res.ravel())
            lgg_best = lgb.LGBMClassifier(**grid_search.best_params_)
        else:
            lgg_best = lgg

        grid_clf_acc = lgg_best.fit(X_train_res, y_train_res)

        if action == 'predict':
            return self._predict(grid_clf_acc)
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, grid_clf_acc.predict(X_valid)),
                                       title=f'LightGBM_GSearch[{gridsearch}]')
            return self._evaluate(grid_clf_acc, X_train_res, X_valid, y_train_res, y_valid)


class EncoderDecoderKNN(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import EncoderDecoderKNN
        >>>
        >>> # To Evaluate
        >>> evaluate_metrics = EncoderDecoderKNN().run("evaluate")
        >>> # To Predict
        >>> prediction = EncoderDecoderKNN().run("predict")
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)

        X_train_ok = X_train_res[y_train_res==0]

        model = self.make_model(X_train_ok)

        model.fit(
            X_train_res,
            X_train_res,
            batch_size=256,
            epochs=100,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode="min")],
            validation_data=(X_train_res, X_train_res))

        enc_all = model.predict(X_train_res)

        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(enc_all, y_train_res)

        if action == 'predict':
            y_test = knn_model.predict(model.predict(self.test_data_preprocessed()))
            results = pd.DataFrame(y_test)
            results.columns = ['Insp']

            return results
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, knn_model.predict(model.predict(X_valid))),
                                       title='EncoderDecoder')
            return self._evaluate(knn_model, model.predict(X_train_res), model.predict(X_valid), y_train_res, y_valid)

    def make_model(self, X_train_ok):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train_ok.shape[-1],)),
            tf.keras.layers.Dense(16, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(X_train_ok.shape[-1], activation='tanh')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError())

        return model


class CatBoost(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode
        self.params = {'iterations': [500],
                      'depth': [6, 8, 10],
                      'learning_rate': [0.01, 0.05, 0.1],
                      'loss_function': ['Logloss', 'CrossEntropy'],
                      'l2_leaf_reg': [3,1,5,10,100],
                      'leaf_estimation_iterations': [10],
                      'logging_level':['Silent'],
                      'random_seed': [42]}

    def run(self, action: str = 'evaluate', gridsearch = False) -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import CatBoost
        >>>
        >>> # To Evaluate
        >>> evaluate_metrics = CatBoost().run("evaluate")
        >>> # To Predict
        >>> prediction = CatBoost().run("predict")
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature, onehot_encode=self.onehot_encode)
        cb = CatBoostClassifier()
        if gridsearch == True:
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                         random_state=1)
            grid_search = GridSearchCV(estimator=cb,
                                       param_grid=self.params, n_jobs=-1, cv=cv,
                                       scoring='roc_auc', error_score=0)
            grid_search.fit(X_train_res, y_train_res.ravel())
            cb_best = CatBoostClassifier(**grid_search.best_params_)
        else:
            cb_best = cb

        cb_best.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(cb)
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, cb_best.predict(X_valid)), title='CatBoostClassifier')
            return self._evaluate(cb_best, X_train_res, X_valid, y_train_res, y_valid)


class SupportVectorMachine(Base):
    def __init__(self, polyfeature=False, onehot_encode=False):
        super().__init__()
        self.polyfeature = polyfeature
        self.onehot_encode = onehot_encode
        self.params = {'C':[1,10,100,1000],
                       'gamma':[1,0.1,0.001,0.0001],
                       'kernel':['linear','rbf']}

    def run(self, action: str = 'evaluate', gridsearch = False):
        """
        >>> from q2_fraud_detection.model import SupportVectorMachine
        >>>
        >>> # To Evaluate
        >>> evaluate_metrics = SupportVectorMachine().run("evaluate")
        >>> # To Predict
        >>> prediction = SupportVectorMachine().run("predict")
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split(polyfeature=self.polyfeature,
                                                                            onehot_encode=self.onehot_encode)
        svc = SVC(kernel='rbf', C=10, gamma=0.1)
        if gridsearch == True:
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                         random_state=1)
            grid_search = GridSearchCV(estimator=svc,
                                       param_grid=self.params, n_jobs=-1, cv=cv,
                                       scoring='roc_auc', error_score=0)
            grid_search.fit(X_train_res, y_train_res.ravel())
            svc_best = SVC(**grid_search.best_params_)
        else:
            svc_best = svc

        svc_best.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(svc_best)
        else:
            self.plot_confusion_matrix(confusion_matrix(y_valid, svc_best.predict(X_valid)), title='SupportVectorMachine')
            return self._evaluate(svc_best, X_train_res, X_valid, y_train_res, y_valid)
