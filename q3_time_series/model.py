from typing import Literal, Union, Dict, List, Any, Tuple
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VAR
import tensorflow as tf

from q3_time_series._base import Base

__all__ = ['VectorAutoRegression', 'StepWiseArima', 'UnivariateMultiStepLSTM']

Countries = Literal["Singapore", "China", "India"]


class VectorAutoRegression(Base):
    def __init__(self):
        super().__init__()

    def run(self, action: str = 'evaluate') -> Union[
        pd.DataFrame, List[Dict[str, Dict[str, Union[Union[float, str], Any]]]]]:
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
            # Use full dataset to get prediction
            model_full = VAR(endog=self.df)
            model_fit_full = model_full.fit()
            return pd.DataFrame(model_fit.forecast(model_fit_full.y, steps=2), index=['2008', '2009'],
                                columns=[self.df.columns])
        else:
            tmp = []
            for col in self.df.columns:
                tmp.append({col: {'rmse_val': sqrt(
                    mean_squared_error(self.valid[col], self._prediction(model_fit, self.valid)[[col]])),
                    'mae_val': mean_absolute_error(self.valid[col],
                                                   self._prediction(model_fit, self.valid)[[col]]),
                    'mape_val': f'{self.mean_absolute_percentage_error(self.valid[col], self._prediction(model_fit, self.valid)[[col]])} %'}})

            return tmp

    @staticmethod
    def _prediction(model, data):
        prediction = model.forecast(model.y, steps=len(data))

        pred = pd.DataFrame(index=range(0, len(prediction)), columns=[data.columns])
        for j in range(0, prediction.shape[1]):
            for i in range(0, len(prediction)):
                pred.iloc[i][j] = prediction[i][j]

        return pred


class StepWiseArima(Base):
    def __init__(self):
        super().__init__()

    def run(self, country: Countries, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[
        Union[Literal["Singapore"], Literal["China"], Literal["India"]], Dict[str, Union[Union[float, str], Any]]]]:
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

        stepwise_model = pm.auto_arima(self.df[country], start_p=1, start_q=1,
                                       max_p=3, max_q=3, m=12,
                                       start_P=0, seasonal=True,
                                       d=1, D=1, trace=True,
                                       error_action='ignore',
                                       suppress_warnings=True,
                                       stepwise=True)
        stepwise_model.fit(self.train[country])
        if action == 'predict':
            # Use full dataset to get prediction
            stepwise_model.fit(self.df[country])
            return pd.DataFrame(stepwise_model.predict(n_periods=2), index=['2008', '2009'], columns=[country])
        else:
            pred_valid = pd.DataFrame(stepwise_model.predict(n_periods=len(self.valid[country])))

            return {country: {'rmse_val': sqrt(mean_squared_error(self.valid[country], pred_valid)),
                              'mae_val': mean_absolute_error(self.valid[country], pred_valid),
                              'mape_val': f'{self.mean_absolute_percentage_error(self.valid[country], pred_valid)} %'}}


class UnivariateMultiStepLSTM(Base):
    def __init__(self, n_steps_in: int, n_steps_out: int):
        super().__init__()
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_features = 1
        self.valid_arb = self.df[int(0.8 * (len(self.df))) - n_steps_in - 1:]

    def run(self, country: Countries, action: str = 'evaluate') -> Union[pd.DataFrame, Dict[
        Union[Literal["Singapore"], Literal["China"], Literal["India"]], Dict[str, Union[Union[float, str], Any]]]]:
        """
        >>> from q3_time_series.model import UnivariateMultiStepLSTM
        >>> # To Evaluate
        >>> evaluate_metrics = UnivariateMultiStepLSTM(3,2).run('Singapore', "evaluate")
        >>> # To Predict
        >>> prediction = UnivariateMultiStepLSTM(3,2).run('Singapore', 'predict')
        """
        assert country in Countries.__args__, \
            f"{country} is not supported, please choose between {Countries.__args__}"
        X_train, y_train = self.split_sequence(self.train[country].values, self.n_steps_in, self.n_steps_out)
        X_valid, y_valid = self.split_sequence(self.valid_arb[country].values, self.n_steps_in, self.n_steps_out)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.n_features))
        X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], self.n_features))

        model = self.make_model()
        model.fit(X_train, y_train, epochs=200, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)],
                  validation_data=(X_valid, y_valid))

        if action == 'predict':
            input = (self.df[country][-self.n_steps_in:].values).reshape((1, self.n_steps_in, self.n_features))
            pred = model.predict(input, verbose=0)

            return pd.DataFrame(pred, columns=['2008', '2009'], index=[country]).T

        else:
            pred_valid = model.predict(X_valid, verbose=0)
            pred_train = model.predict(X_train, verbose=0)

            return {country: {'rmse_train': sqrt(mean_squared_error([y_train[i][0] for i in range(0, len(y_train))],
                                                                    [pred_train[i][0] for i in
                                                                     range(0, len(pred_train))])),
                              'rmse_val': sqrt(mean_squared_error([y_valid[i][0] for i in range(0, len(y_valid))],
                                                                  [pred_valid[i][0] for i in
                                                                   range(0, len(pred_valid))])),
                              'mae_train': mean_absolute_error([y_train[i][0] for i in range(0, len(y_train))],
                                                               [pred_train[i][0] for i in range(0, len(pred_train))]),
                              'mae_val': mean_absolute_error([y_valid[i][0] for i in range(0, len(y_valid))],
                                                             [pred_valid[i][0] for i in range(0, len(pred_valid))]),
                              'mape_train': f'{self.mean_absolute_percentage_error([y_train[i][0] for i in range(0, len(y_train))], [pred_train[i][0] for i in range(0, len(pred_train))])} %',
                              'mape_val': f'{self.mean_absolute_percentage_error([y_valid[i][0] for i in range(0, len(y_valid))], [pred_valid[i][0] for i in range(0, len(pred_valid))])} %'}}

    def make_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, activation='relu', input_shape=(self.n_steps_in, self.n_features)),
            tf.keras.layers.Dense(self.n_steps_out)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError())

        return model

    @staticmethod
    def split_sequence(sequence, n_steps_in: int, n_steps_out: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


class MultivariateMultiStepLSTM(Base):
    def __init__(self, n_steps_in: int, n_steps_out: int):
        super().__init__()
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.valid_arb = self.df[int(0.8 * (len(self.df))) - n_steps_in - 1:]

    def run(self):
        """
        For experimental purpose, since the evaluation result is bad, this model will not be use for prediction
        >>> from q3_time_series.model import MultivariateMultiStepLSTM
        >>> # To Evaluate
        >>> evaluate_metrics = MultivariateMultiStepLSTM(3,2).run()
        """
        dataset_train = np.hstack(self.hstacK_generator(self.train))
        dataset_valid = np.hstack(self.hstacK_generator(self.valid_arb))

        X_train, y_train = self.split_sequences(dataset_train, self.n_steps_in, self.n_steps_out)
        X_valid, y_valid = self.split_sequences(dataset_valid, self.n_steps_in, self.n_steps_out)

        model = self.make_model(X_train.shape[2])
        model.fit(X_train, y_train, epochs=200, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)],
                  validation_data=(X_valid, y_valid))

        pred_valid = model.predict(X_valid, verbose=0)
        pred_train = model.predict(X_train, verbose=0)

        tmp = []
        for j, col in enumerate(self.df.columns):
            tmp.append({col: {'rmse_train': sqrt(mean_squared_error([y_train[i][0][j] for i in range(0, len(y_train))],
                                                                    [pred_train[i][0][j] for i in
                                                                     range(0, len(pred_train))])),
                              'rmse_val': sqrt(mean_squared_error([y_valid[i][0][j] for i in range(0, len(y_valid))],
                                                                  [pred_valid[i][0][j] for i in
                                                                   range(0, len(pred_valid))])),
                              'mae_train': mean_absolute_error([y_train[i][0][j] for i in range(0, len(y_train))],
                                                               [pred_train[i][0][j] for i in
                                                                range(0, len(pred_train))]),
                              'mae_val': mean_absolute_error([y_valid[i][0][j] for i in range(0, len(y_valid))],
                                                             [pred_valid[i][0][j] for i in range(0, len(pred_valid))]),
                              'mape_train': f'{self.mean_absolute_percentage_error([y_train[i][0][j] for i in range(0, len(y_train))], [pred_train[i][0][j] for i in range(0, len(pred_train))])} %',
                              'mape_val': f'{self.mean_absolute_percentage_error([y_valid[i][0][j] for i in range(0, len(y_valid))], [pred_train[i][0][j] for i in range(0, len(pred_valid))])} %'}})

        return tmp

    def make_model(self, n_features: int):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(200, activation='relu', input_shape=(self.n_steps_in, n_features)),
            tf.keras.layers.RepeatVector(self.n_steps_out),
            tf.keras.layers.LSTM(200, activation='relu', return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features)),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError())

        return model

    def hstacK_generator(self, df) -> Tuple[Any, ...]:
        tmp = []
        for country in self.df.columns:
            seq = df[country].values.reshape((len(df[country]), 1))
            tmp.append(seq)

        return tuple(tmp)

    @staticmethod
    def split_sequences(sequences, n_steps_in: int, n_steps_out: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)
