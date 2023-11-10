from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LightGBMModel:
    def __init__(
        self,
        preprocessed_data: pd.DataFrame = pd.DataFrame(),
        output: str = 'clock_hours',
        features_excluded: List[str] = ['operation_key', 'order_number'],
        model_params: Optional[Dict] = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "random_state": 42
        },
        grid_search_flag: bool = True,
        grid_search_parameters: Optional[Dict] = {
            "num_leaves": [31, 50, 100],
            "learning_rate": [0.05, 0.1, 0.2],
            "n_estimators": [100, 200, 300]
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
        self.label_encoders = {}  # Dictionary to store label encoders for categorical variables

    def _metrics_calculation(self, true_values, predicted_values):
        """Calculate Error Metrics

        Args:
            true_values: True values for the output variable
            predicted_values: Values that the model predicted for the output variable
        Returns:
            Dict of error metrics
        """
        MAE = mean_absolute_error(true_values, predicted_values)
        MSE = mean_squared_error(true_values, predicted_values)
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

        # Encode categorical variables using label encoders
        for col in X.columns:
            if X[col].dtype == 'O':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size
        )

        return X_train, X_test, y_train, y_test

    def fit(self):
        """Trains the LightGBM Regressor model on training data, uses grid search to find the best parameters.

        Returns:
            Regressor with the best fit parameters
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_operation_data(self.preprocessed_data)

        if self.grid_search_flag:
            reg = lgb.LGBMRegressor(**self.model_params)
            search = GridSearchCV(
                reg,
                param_grid=self.grid_search_parameters,
                cv=self.cv_splits
            )
            search.fit(X=self.X_train, y=self.y_train)
            reg_bestfit = search.best_estimator_
        else:
            reg_bestfit = lgb.LGBMRegressor(**self.model_params)
            reg_bestfit.fit(self.X_train, self.y_train)

        self.best_estimator = reg_bestfit

        y_preds = reg_bestfit.predict(self.X_test)

        metrics = self._metrics_calculation(self.y_test, y_preds)

        logger = logging.getLogger(__name__)
        logger.info(f"Model includes the following features: {self.X_train.columns}")
        logger.info(f"Model uses the following parameters: {reg_bestfit.get_params()}")
        logger.info(f"Model's test scores: {metrics}")

        return reg_bestfit

    def predict(self, pred_preprocessed_data):
        """Predict work duration for input list of order numbers.

        Args:
            pred_preprocessed_data: Preprocessed data containing features for prediction
        Returns:
            Dataframe with order number, operation key, and predicted values
        """
        if self.best_estimator is None:
            print('Please fit the model before predicting')
            return  # Return without making predictions if the model is not fitted

        # Handle categorical variables in the prediction data using label encoders from training
        for col in pred_preprocessed_data.columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                pred_preprocessed_data[col] = le.transform(pred_preprocessed_data[col])

        X_pred = pred_preprocessed_data.drop(self.features_excluded, axis=1)
        y_preds = self.best_estimator.predict(X_pred)

        final_output = pd.DataFrame({
            'order_number': pred_preprocessed_data['order_number'],
            'operation_key': pred_preprocessed_data['operation_key'],
            'predicted_values': y_preds
        })

        return final_output
