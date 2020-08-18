import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, f1_score

__all__ = ['Base']


class Base:
    def __init__(self):
        self.df = pd.read_csv('data/cleanded_dataset.csv')
        self.train = self.df[self.df['Insp'] != 'unkn']
        self.test = self.df[self.df['Insp'] == 'unkn']

    def _train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Sanity check before spliting into 80% training and 20% validation set """
        X = self.train[['ID', 'Prod', 'Quant', 'Val']]
        y = self.train['Insp']

        # Converting float to categorical variables
        X['ID'] = X['ID'].astype('int')
        X['Prod'] = X['Prod'].astype('int')

        # Converting string to machine readable binary
        mapper = {'ok': 0, 'fraud': 1}
        y = y.map(mapper)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

        return X_train_res, X_valid, y_train_res, y_valid

    def test_data_preprocessed(self) -> pd.DataFrame:
        """ Converting testing dataset to training dataset format """
        X_test = self.test.drop(columns=['Insp'])

        X_test['ID'] = X_test['ID'].astype('int')
        X_test['Prod'] = X_test['Prod'].astype('int')

        return X_test

    def _predict(self, model) -> pd.DataFrame:
        y_test = model.predict(self.test_data_preprocessed())
        results = pd.DataFrame(y_test)
        results.columns = ['Insp']

        return results

    @staticmethod
    def _evaluate(model, X_train_res:pd.DataFrame, X_valid: pd.DataFrame, y_train_res: pd.DataFrame
                  , y_valid: pd.DataFrame, threshold = None) -> Dict[str, float]:
        y_train_pre = model.predict(X_train_res)
        y_valid_pre = model.predict(X_valid)

        if isinstance(threshold, float):
            return {'auc_train': roc_auc_score(y_train_res, y_train_pre > threshold),
                'auc_valid': roc_auc_score(y_valid, y_valid_pre > threshold),
                'acc_train': accuracy_score(y_train_res, y_train_pre > threshold),
                'acc_valid': accuracy_score(y_valid, y_valid_pre > threshold),
                'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre > threshold),
                'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre > threshold),
                'f1_score_train': f1_score(y_train_res, y_train_pre > threshold),
                'f1_score_valid': f1_score(y_valid, y_valid_pre > threshold)
                }

        return {'auc_train': roc_auc_score(y_train_res, y_train_pre),
                'auc_valid': roc_auc_score(y_valid, y_valid_pre),
                'acc_train': accuracy_score(y_train_res, y_train_pre),
                'acc_valid': accuracy_score(y_valid, y_valid_pre),
                'matthew_corr_train': matthews_corrcoef(y_train_res, y_train_pre),
                'matthew_corr_valid': matthews_corrcoef(y_valid, y_valid_pre),
                'f1_score_train': f1_score(y_train_res, y_train_pre),
                'f1_score_valid': f1_score(y_valid, y_valid_pre)
                }

    @staticmethod
    def plot_confusion_matrix(cm, classes=None,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if classes is None:
            classes = ['ok', 'fraud']
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)


        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
