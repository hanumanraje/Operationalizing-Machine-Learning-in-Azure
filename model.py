# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 21:24:32 2023

@author: Raje
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class RandomForestModel:
    def __init__(
        self,
        preprocessed_data: pd.DataFrame = pd.DataFrame(),
        output: str = 'clock_hours',
        date_cutoff: str = '2023-02-20',
        features_excluded: List[str] = ['operation_key', 'order_number'],
        model_params: Optional[Dict] = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        grid_search_flag: bool = True,
        grid_search_parameters: Optional[Dict] = {
            "n_estimators": [50, 100, 200],
            "max_depth": [8, 10, 12]
        },
        scoring_metrics: List[str] = ["MAE", "MSE", "RMSE"],
        test_size: float = 0.2,
        cv_splits: int = 3,
        forecasting: Dict = {
            "order_number_operation_key": [[40411022, 80], [35396175, 60]],
            "pred_threshold": 1
        }
    ) -> None:
        self.preprocessed_data = preprocessed_data
        self.output = output
        self.date_cutoff = date_cutoff
        self.features_excluded = features_excluded
        self.model_params = model_params
        self.grid_search_flag = grid_search_flag
        self.grid_search_parameters = grid_search_parameters
        self.scoring_metrics = scoring_metrics
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.forecasting = forecasting

        self.X_train = np.array([])
        self.X_test = np.array([])
        self.y_train = []
        self.y_test = []
        self.best_estimator = None
        self.shap_values = []

    def _metrics_calculation(self, true_values, predicted_values):
        """Calculate Error Metrics

        Args:
            true_values: True values for the output variable
            predicted_values: Values that the model predicted for the output variable
        Returns:
            Dict of error metrics
        """
        true_values, predicted_values = np.array(true_values), np.array(predicted_values)
        MAE = np.mean(np.abs(true_values - predicted_values))
        MSE = np.mean((true_values - predicted_values) ** 2)
        RMSE = np.sqrt(MSE)
        metrics = {'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE}

        return {key: metrics[key] for key in self.scoring_metrics}

    def _split_operation_data(self, preprocessed_data: pd.DataFrame) -> Tuple:
        """Splits data into features and target training and test sets.

        Args:
            preprocessed_data: Preprocessed data containing features and target
        Returns:
            Split data.
        """
        X = preprocessed_data.drop(self.features_excluded, axis=1).drop(self.output, axis=1)
        y = preprocessed_data[self.output]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size
        )

        return X_train, X_test, y_train, y_test

    def fit(self):
        """Trains the Random Forest Regressor model on training data, uses grid search to find the best parameters.

        Returns:
            Regressor with the best fit parameters
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_operation_data(self.preprocessed_data)

        # Handle categorical variables with LabelEncoder
        categorical_columns = [col for col in self.X_train.columns if self.X_train[col].dtype == 'O']
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            self.X_train[col] = le.fit_transform(self.X_train[col])
            self.X_test[col] = le.transform(self.X_test[col])
            label_encoders[col] = le

        if self.grid_search_flag:
            reg = RandomForestRegressor(**self.model_params)
            search = GridSearchCV(
                reg,
                param_grid=self.grid_search_parameters,
                cv=self.cv_splits
            )
            search.fit(X=self.X_train, y=self.y_train)
            reg_bestfit = search.best_estimator_
        else:
            reg_bestfit = RandomForestRegressor(**self.model_params)
            reg_bestfit.fit(self.X_train, self.y_train)

        self.best_estimator = reg_bestfit

        y_preds = reg_bestfit.predict(self.X_test)
        y_preds = [i if i > 0 else 0 for i in y_preds]

        metrics = self._metrics_calculation(self.y_test, y_preds)

        logger = logging.getLogger(__name__)
        logger.info(f"Model includes the following features: {self.X_train.columns}")
        logger.info(f"Model uses the following parameters: {reg_bestfit.get_params()}")
        logger.info(f"Model's test scores: {metrics}")

        explainer = shap.Explainer(reg_bestfit)
        self.shap_values = explainer.shap_values(pd.DataFrame(self.X_test, columns=self.X_test.columns))

        return reg_bestfit, metrics

    def shap_plot(self):
        """Create and store a plot for feature importances using SHAP values.

        Returns:
            Plot
        """
        shap.summary_plot(self.shap_values, self.X_test, show=False)

        return plt

    def predict(self, pred_preprocessed_data, op_data_pred, pred_threshold, prep_models):
        """Predict work duration for input list of order numbers and adjust predictions based on pred_threshold.

        Args:
            pred_preprocessed_data: Preprocessed data containing features and target for prediction
            op_data_pred: Raw pre-scaled operation data
            pred_threshold: The percentage of difference to adjust based on SAP prediction
            prep_models: Dict to save/load fitted preprocessing models
        Returns:
            Dataframe with order number, operation key, non-adjusted prediction, SAP prediction, and adjusted predictions
        """
        if self.best_estimator is None:
            print('Please fit the model before predicting')
        else:
            # Handle categorical variables in the prediction data
            label_encoders_pred = {}
            for col in categorical_columns:
                le = label_encoders.get(col, LabelEncoder())
                pred_preprocessed_data[col] = le.transform(pred_preprocessed_data[col])
                label_encoders_pred[col] = le

            X_pred = pred_preprocessed_data.drop(self.output, axis=1).drop(self.features_excluded, axis=1)
            y_preds = self.best_estimator.predict(X_pred)

            if "fitted_scaler" in prep_models:
                pred_preprocessed_data['key'] = pred_preprocessed_data['order_number'].astype(int).astype(str) + \
                                                '-' + pred_preprocessed_data['operation_key'].astype(int).astype(str)
                pred_preprocessed_data.drop(columns=['forecast_man_hours'], inplace=True)
                pred_preprocessed_data = pred_preprocessed_data.merge(op_data_pred[['key','forecast_man_hours']], how='left', on='key')
                sap_preds = list(pred_preprocessed_data['forecast_man_hours'])
            else:
                sap_preds = list(pred_preprocessed_data['forecast_man_hours'])

            y_preds = y_preds * 2

            y_preds_adj = [sap_preds[i] + (y_preds[i] - sap_preds[i]) * pred_threshold for i in range(len(y_preds))]

            final_output = pd.DataFrame({
                'order_number': pred_preprocessed_data['order_number'],
                'operation_key': pred_preprocessed_data['operation_key'],
                'SAP_pred': sap_preds,
                'non_adjusted_pred': y_preds,
                'adjusted_pred': y_preds_adj
            })

            final_output['non_adjusted_pred'] = round(final_output['non_adjusted_pred'], 1)
            final_output['adjusted_pred'] = round(final_output['adjusted_pred'], 1)

            return final_output
