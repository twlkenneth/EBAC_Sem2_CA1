import pandas as pd
import numpy as np

__all__ = ['Base']


class Base:
    def __init__(self):
        self.df = pd.read_csv('data/cleaned_data.csv').set_index('Year')
        self.train = self.df[:int(0.8 * (len(self.df)))]
        self.valid = self.df[int(0.8 * (len(self.df))):]

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
