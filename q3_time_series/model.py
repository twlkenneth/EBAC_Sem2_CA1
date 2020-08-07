from typing import Literal
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VAR

from q3_time_series.base import Base

__all__ = ['VectorAutoRegression', 'StepWiseArima']

Countries = Literal["Singapore", "China", "India"]


class VectorAutoRegression(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate'):
        """
        >>> from q3_time_series.model import VectorAutoRegression
        >>> # To Evaluate
        >>> evaluate_metrics = VectorAutoRegression().run("evaluate")
        >>> # To Predict
        >>> prediction = VectorAutoRegression().run("predict")
        """
        model = VAR(endog=self.train)
        model_fit = model.fit()
        if action == 'predict':
            return pd.DataFrame(model_fit.forecast(model_fit.y, steps=2),index=['2008', '2009'] ,columns=[self.df.columns])
        else:
            prediction = model_fit.forecast(model_fit.y, steps=len(self.valid))

            pred = pd.DataFrame(index=range(0,len(prediction)),columns=[self.df.columns])
            for j in range(0,prediction.shape[1]):
                for i in range(0, len(prediction)):
                   pred.iloc[i][j] = prediction[i][j]

            tmp = []
            for col in self.df.columns:
                tmp.append({col:{'rmse_val' : sqrt(mean_squared_error(self.valid[col], pred[[col]])),
                                 'mae_val' : mean_absolute_error(self.valid[col], pred[[col]]),
                                 'mape_val': f'{self.mean_absolute_percentage_error(self.valid[col], pred[[col]])} %'}})

            return tmp


class StepWiseArima(Base):
    def __init__(self):
        super().__init__()

    def run(self, country: Countries, action: str = 'evaluate'):
        """
        :param country: type string in ["Singapore", "China", "India"]
        :param action: type string "evaluate" or "predict"
        :return: type pd.DataFrame for "predict" or type Dict for "evaluate"

        >>> from q3_time_series.model import StepWiseArima
        >>> # To Evaluate
        >>> evaluate_metrics = StepWiseArima().run("Singapore", "evaluate")
        >>> # To Predict
        >>> prediction = StepWiseArima().run("Singapore", "predict")
        """
        assert country in Countries.__args__, \
            f"{country} is not supported, please choose between {Countries.__args__}"

        stepwise_model = pm.auto_arima( self.df[country], start_p=1, start_q=1,
                                        max_p = 3, max_q = 3, m = 12,
                                        start_P = 0, seasonal = True,
                                        d = 1, D = 1, trace = True,
                                        error_action = 'ignore',
                                        suppress_warnings = True,
                                        stepwise = True)
        stepwise_model.fit(self.train[country])
        if action == 'predict':
            return pd.DataFrame(stepwise_model.predict(n_periods=2), index=['2008', '2009'], columns=[country])
        else:
            pred = pd.DataFrame(stepwise_model.predict(n_periods=len(self.valid[country])))

            return {country: {'rmse_val' : sqrt(mean_squared_error(self.valid[country], pred)),
                                'mae_val' : mean_absolute_error(self.valid[country], pred),
                                'mape_val': f'{self.mean_absolute_percentage_error(self.valid[country], pred)} %'}}
