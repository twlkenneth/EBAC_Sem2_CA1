import numpy as np
import pandas as pd
from typing import Dict, Union

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import xgboost as xgb

from q2_fraud_detection.base import Base

__all__ = ['LRegression', 'DecisionTree', 'NaiveBayesClassifier', 'RandomForest', 'XGBoost', 'TensorflowMLP',
           'LightGBM', 'EncoderDecoderKNN']


class LRegression(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        """
        >>> from q2_fraud_detection.model import LRegression
        >>>
        >>> # To Evaluate (same for all other models)
        >>> evaluate_metrics = LRegression().run("evaluate")
        >>> # To Predict (same for all other models)
        >>> prediction = LRegression().run("predict")
        """
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
        parameters = {
            'C': np.linspace(1, 10, 10)
        }
        lr = LogisticRegression()
        clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
        clf.fit(X_train_res, y_train_res.ravel())

        lr1 = LogisticRegression(C=clf.best_params_['C'], penalty='l2', verbose=5)
        lr1.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(lr1)
        else:
            return self._evaluate(lr1, X_train_res, X_valid, y_train_res, y_valid)


class DecisionTree(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(clf)
        else:
            return self._evaluate(clf, X_train_res, X_valid, y_train_res, y_valid)


class NaiveBayesClassifier(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
        nbc = GaussianNB()
        nbc.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(nbc)
        else:
            return self._evaluate(nbc, X_train_res, X_valid, y_train_res, y_valid)


class RandomForest(Base):
    def __init__(self):
        super().__init__()
        self.param_grid =  {'n_estimators': [200, 500],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'max_depth' : [4,5,6,7,8],
                            'criterion' :['gini', 'entropy']}

    def run(self, action: str = 'evaluate', gridsearch = False) -> Union[pd.DataFrame, Dict[str, float]]:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()

        rf = RandomForestClassifier(random_state=0)
        if gridsearch == True:
            grid_rf = GridSearchCV(estimator=rf, cv=5, param_grid=self.param_grid, scoring='roc_auc')
        else:
            grid_rf = rf

        grid_rf.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            return self._predict(grid_rf)
        else:
            return self._evaluate(grid_rf, X_train_res, X_valid, y_train_res, y_valid)


class XGBoost(Base):
    def __init__(self):
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

    def run(self, action: str = 'evaluate', gridsearch = False) -> Union[pd.DataFrame, Dict[str, float]]:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
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
    def __init__(self):
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

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
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
    def __init__(self):
        super().__init__()
        self.class_weight = [None, 'balanced']
        self.boosting_type = ['gbdt', 'goss', 'dart']
        self.num_leaves = [30, 50, 100, 150]
        self.learning_rate = list(np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=10))
        self.lgg_grid = dict(class_weight=self.class_weight, boosting_type=self.boosting_type,
                             num_leaves=self.num_leaves,
                             learning_rate=self.learning_rate)

    def run(self, action: str = 'evaluate', gridsearch = False) -> Union[pd.DataFrame, Dict[str, float]]:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()

        lgg = lgb.LGBMClassifier()
        if gridsearch == True:
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3,
                                         random_state=1)
            grid_search = GridSearchCV(estimator=lgg,
                                       param_grid=self.lgg_grid, n_jobs=-1, cv=cv,
                                       scoring='roc_auc', error_score=0)
        else:
            grid_search = lgg

        grid_clf_acc = grid_search.fit(X_train_res, y_train_res)

        if action == 'predict':
            return self._predict(grid_clf_acc)
        else:
            return self._evaluate(grid_clf_acc, X_train_res, X_valid, y_train_res, y_valid)


class EncoderDecoderKNN(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[str, float]]:
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()

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
