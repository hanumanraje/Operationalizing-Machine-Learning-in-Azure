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
    self.X_train, self.X_test, self.y_train, self.y_test = self._split_operation_data(self.preprocessed_data)

    lgb_train = lgb.Dataset(self.X_train, self.y_train)
    lgb_eval = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)

    reg = lgb.train(
        self.model_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=100,
        early_stopping_rounds=50,
        num_boost_round=10000
    )

    self.best_estimator = reg

    y_preds = reg.predict(self.X_test)

    metrics = self._metrics_calculation(self.y_test, y_preds)

    logger = logging.getLogger(__name__)
    logger.info(f"Model includes the following features: {list(self.X_train.columns)}")
    logger.info(f"Model's test scores: {metrics}")

    explainer = shap.Explainer(reg)
    self.shap_values = explainer.shap_values(pd.DataFrame(self.X_test, columns=self.X_train.columns))

    logger.info(f"Returning from fit: reg = {reg}, metrics = {metrics}")
    return reg, metrics


def predict(self, pred_preprocessed_data, op_data_pred, pred_threshold, prep_models):
    """Predict work duration for input data and adjust predictions based on pred_threshold.

    Args:
        pred_preprocessed_data: Preprocessed data containing features for prediction.
        op_data_pred: Raw pre-scaled operation data.
        pred_threshold: The percentage of difference to adjust based on SAP prediction.
        prep_models: Dict to save/load fitted preprocessing models.

    Returns:
        Dataframe with order number, operation key, non-adjusted prediction, SAP prediction, and adjusted predictions.
    """
    if self.best_estimator is None:
        print('Please fit the model before predicting')
        return None

    # Handle categorical variables in the prediction data using label encoders from training
    for col in pred_preprocessed_data.columns:
        if col in self.label_encoders:
            le = self.label_encoders[col]
            pred_preprocessed_data[col] = le.transform(pred_preprocessed_data[col])

    X_pred = pred_preprocessed_data.drop(self.features_excluded, axis=1)
    y_preds = self.best_estimator.predict(X_pred)

    if "fitted_scaler" in prep_models:
        pred_preprocessed_data['key'] = pred_preprocessed_data['order_number'].astype(int).astype(str) + \
                                        '-' + pred_preprocessed_data['operation_key'].astype(int).astype(str)
        pred_preprocessed_data.drop(columns=['forecast_man_hours'], inplace=True)
        pred_preprocessed_data = pred_preprocessed_data.merge(op_data_pred[['key','forecast_man_hours']], how='left', on='key')
        sap_preds = list(pred_preprocessed_data['forecast_man_hours'])
    else:
        sap_preds = list(pred_preprocessed_data['forecast_man_hours'])

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

