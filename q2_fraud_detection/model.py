import numpy as np
import pandas as pd
from typing import Dict

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import xgboost as xgb

from q2_fraud_detection.base import Base

__all__ = ['LRegression', 'DecisionTree', 'NaiveBayesClassifier', 'RandomForest', 'XGBoost', 'TensorflowANN',
           'LightGBM']


class LRegression(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate'):
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
            y_test = lr1.predict(self.test_data_preprocessed())
            results = pd.DataFrame(y_test)
            results.columns = ['Insp']

            return results
        else:
            y_train_pre = lr1.predict(X_train_res)
            y_valid_pre = lr1.predict(X_valid)

            return {'auc_train': roc_auc_score(y_train_res, y_train_pre),
                    'auc_valid': roc_auc_score(y_valid, y_valid_pre),
                    'acc_train': accuracy_score(y_train_res, y_train_pre),
                    'acc_valid': accuracy_score(y_valid, y_valid_pre),
                    'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre),
                    'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre),
                    'f1_score_train': f1_score(y_train_res, y_train_pre),
                    'f1_score_valid': f1_score(y_valid, y_valid_pre)
                    }


class DecisionTree(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate'):
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            y_test = clf.predict(self.test_data_preprocessed())
            results = pd.DataFrame(y_test)
            results.columns = ['Insp']

            return results
        else:
            y_train_pre = clf.predict(X_train_res)
            y_valid_pre = clf.predict(X_valid)

            return {'auc_train': roc_auc_score(y_train_res, y_train_pre),
                    'auc_valid': roc_auc_score(y_valid, y_valid_pre),
                    'acc_train': accuracy_score(y_train_res, y_train_pre),
                    'acc_valid': accuracy_score(y_valid, y_valid_pre),
                    'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre),
                    'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre),
                    'f1_score_train': f1_score(y_train_res, y_train_pre),
                    'f1_score_valid': f1_score(y_valid, y_valid_pre)}


class NaiveBayesClassifier(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate'):
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
        nbc = GaussianNB()
        nbc.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            y_test = nbc.predict(self.test_data_preprocessed())
            results = pd.DataFrame(y_test)
            results.columns = ['Insp']

            return results
        else:
            y_train_pre = nbc.predict(X_train_res)
            y_valid_pre = nbc.predict(X_valid)

            return {'auc_train': roc_auc_score(y_train_res, y_train_pre),
                    'auc_valid': roc_auc_score(y_valid, y_valid_pre),
                    'acc_train': accuracy_score(y_train_res, y_train_pre),
                    'acc_valid': accuracy_score(y_valid, y_valid_pre),
                    'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre),
                    'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre),
                    'f1_score_train': f1_score(y_train_res, y_train_pre),
                    'f1_score_valid': f1_score(y_valid, y_valid_pre)}


class RandomForest(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate'):
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
        rf = RandomForestClassifier(random_state=0)
        rf.fit(X_train_res, y_train_res.ravel())

        if action == 'predict':
            y_test = rf.predict(self.test_data_preprocessed())
            results = pd.DataFrame(y_test)
            results.columns = ['Insp']

            return results
        else:
            y_train_pre = rf.predict(X_train_res)
            y_valid_pre = rf.predict(X_valid)

            return {'auc_train': roc_auc_score(y_train_res, y_train_pre),
                    'auc_valid': roc_auc_score(y_valid, y_valid_pre),
                    'acc_train': accuracy_score(y_train_res, y_train_pre),
                    'acc_valid': accuracy_score(y_valid, y_valid_pre),
                    'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre),
                    'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre),
                    'f1_score_train': f1_score(y_train_res, y_train_pre),
                    'f1_score_valid': f1_score(y_valid, y_valid_pre)}


class XGBoost(Base):
    def __init__(self):
        super().__init__()
        self.space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
                      'gamma': hp.uniform('gamma', 1, 9),
                      'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
                      'reg_lambda': hp.uniform('reg_lambda', 0, 1),
                      'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                      'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                      'n_estimators': 180,
                      'seed': 0
                      }

    def run(self, action: str = 'evaluate'):
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
        trials = Trials()
        best_hyperparams = fmin(fn=self.objective,
                                space=self.space,
                                algo=tpe.suggest,
                                max_evals=100,
                                trials=trials)
        clf_xgb = xgb.XGBClassifier(max_depth=int(best_hyperparams['max_depth']), gamma=best_hyperparams['gamma'],
                                    reg_alpha=int(best_hyperparams['reg_alpha']),
                                    min_child_weight=int(best_hyperparams['min_child_weight']),
                                    reg_lambda=best_hyperparams['reg_lambda'],
                                    colsample_bytree=best_hyperparams['colsample_bytree'])
        clf_xgb.fit(X_train_res, y_train_res)

        if action == 'predict':
            y_test = clf_xgb.predict(self.test_data_preprocessed())
            results = pd.DataFrame(y_test)
            results.columns = ['Insp']

            return results
        else:
            y_train_pre = clf_xgb.predict(X_train_res)
            y_valid_pre = clf_xgb.predict(X_valid)

            return {'auc_train': roc_auc_score(y_train_res, y_train_pre),
                    'auc_valid': roc_auc_score(y_valid, y_valid_pre),
                    'acc_train': accuracy_score(y_train_res, y_train_pre > 0.5),
                    'acc_valid': accuracy_score(y_valid, y_valid_pre > 0.5),
                    'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre > 0.5),
                    'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre > 0.5),
                    'f1_score_train': f1_score(y_train_res, y_train_pre > 0.5),
                    'f1_score_valid': f1_score(y_valid, y_valid_pre > 0.5)
                    }

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


class TensorflowANN(Base):
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

    def run(self, action: str = 'evaluate'):
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
            y_test = model.predict(self.test_data_preprocessed())
            results = pd.DataFrame(y_test)
            results.columns = ['Insp']

            return results
        else:
            y_train_pre = model.predict(X_train_res)
            y_valid_pre = model.predict(X_valid)

            return {'auc_train': roc_auc_score(y_train_res, y_train_pre),
                    'auc_valid': roc_auc_score(y_valid, y_valid_pre),
                    'acc_train': accuracy_score(y_train_res, y_train_pre > 0.5),
                    'acc_valid': accuracy_score(y_valid, y_valid_pre > 0.5),
                    'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre > 0.5),
                    'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre > 0.5),
                    'f1_score_train': f1_score(y_train_res, y_train_pre > 0.5),
                    'f1_score_valid': f1_score(y_valid, y_valid_pre > 0.5)
                    }

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

    def run(self, action: str = 'evaluate'):
        X_train_res, X_valid, y_train_res, y_valid = self._train_test_split()
        lgg = lgb.LGBMClassifier()
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3,
                                     random_state=1)
        grid_search = GridSearchCV(estimator=lgg,
                                   param_grid=self.lgg_grid, n_jobs=-1, cv=cv,
                                   scoring='roc_auc', error_score=0)
        grid_clf_acc = grid_search.fit(X_train_res, y_train_res)

        if action == 'predict':
            y_test = grid_clf_acc.predict(self.test_data_preprocessed())
            results = pd.DataFrame(y_test)
            results.columns = ['Insp']

            return results
        else:
            y_train_pre = grid_clf_acc.predict(X_train_res)
            y_valid_pre = grid_clf_acc.predict(X_valid)

            return {'auc_train': roc_auc_score(y_train_res, y_train_pre),
                    'auc_valid': roc_auc_score(y_valid, y_valid_pre),
                    'acc_train': accuracy_score(y_train_res, y_train_pre > 0.5),
                    'acc_valid': accuracy_score(y_valid, y_valid_pre > 0.5),
                    'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre > 0.5),
                    'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre > 0.5),
                    'f1_score_train': f1_score(y_train_res, y_train_pre > 0.5),
                    'f1_score_valid': f1_score(y_valid, y_valid_pre > 0.5)
                    }
