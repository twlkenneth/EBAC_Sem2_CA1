import pandas as pd
from typing import Tuple

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

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
