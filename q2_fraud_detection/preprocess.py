import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self):
        self.df = pd.read_csv('data/reports.csv')

    def preprocess(self) -> pd.DataFrame:
        """ first step data cleaning """
        df = self.df
        df['ID'] = self.label_encoder(df['ID'])
        df['Prod'] = self.label_encoder(df['Prod'])
        y = df['Insp']
        X = df[['ID', 'Prod', 'Quant', 'Val']]
        df_cleaned = pd.concat([self.missing_data_imputer(X), y], axis =1)

        return df_cleaned

    @staticmethod
    def missing_data_imputer(X: pd.DataFrame) -> pd.DataFrame:
        """ default n=5 for KNN Imputer """
        imputer = KNNImputer()
        imputer.fit(X)
        X_transform = imputer.transform(X)
        df_temp = pd.DataFrame(X_transform)
        df_temp.columns = X.columns

        return df_temp

    @staticmethod
    def label_encoder(x: pd.Series) -> pd.Series:
        """ to convert categorical strings to numerical integer """
        le = LabelEncoder()
        le.fit(x)
        return le.transform(x)
