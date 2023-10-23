import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class NeuralNetworkModel:
    def __init__(
        self,
        preprocessed_data: pd.DataFrame = pd.DataFrame(),
        output: str = 'clock_hours',
        date_cutoff: str = '2023-02-20',
        features_excluded: List[str] = ['operation_key', 'order_number'],
        model_params: Optional[Dict] = {
            "hidden_layers": [64, 32],
            "activation": "relu",
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            "early_stopping_patience": 10,
            "random_state": 42
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
        self.scoring_metrics = scoring_metrics
        self.test_size = test_size
        self.cv_splits = cv_splits
        self.forecasting = forecasting

        self.X_train = np.array([])
        self.X_test = np.array([])
        self.y_train = []
        self.y_test = []
        self.model = None
        self.history = None
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
            X, y, test_size=self.test_size, random_state=self.model_params["random_state"]
        )

        return X_train, X_test, y_train, y_test

    def _build_neural_network(self):
        """Builds a neural network model."""
        model = Sequential()
        model.add(Dense(units=self.model_params["hidden_layers"][0], input_dim=self.X_train.shape[1], activation=self.model_params["activation"]))
        
        for units in self.model_params["hidden_layers"][1:]:
            model.add(Dense(units=units, activation=self.model_params["activation"]))

        model.add(Dense(units=1))  # Output layer with 1 neuron for regression
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.model_params["learning_rate"]))

        return model

    def fit(self):
        """Trains the neural network model on training data.

        Returns:
            Fitted neural network model.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_operation_data(self.preprocessed_data)

        self.model = self._build_neural_network()

        # Implement early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.model_params["early_stopping_patience"], verbose=1, restore_best_weights=True)

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.model_params["epochs"],
            batch_size=self.model_params["batch_size"],
            callbacks=[early_stopping],
            verbose=1
        )

        return self.model

    def predict(self, pred_preprocessed_data, op_data_pred, pred_threshold, prep_models):
        """Predict work duration for input list of order numbers.

        Args:
            pred_preprocessed_data: Preprocessed data containing features for prediction
            op_data_pred: Raw pre-scaled operation data
            pred_threshold: The percentage of difference to adjust based on SAP prediction
            prep_models: Dict to save/load fitted preprocessing models
        Returns:
            Dataframe with order number, operation key, non-adjusted prediction, SAP prediction, and adjusted predictions
        """
        if self.model is None:
            print('Please fit the model before predicting')
            return  # Return without making predictions if the model is not fitted

        # Handle categorical variables in the prediction data using label encoders from training
        for col in pred_preprocessed_data.columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                pred_preprocessed_data[col] = le.transform(pred_preprocessed_data[col])

        y_preds = self.model.predict(pred_preprocessed_data)

        if "fitted_scaler" in prep_models:
            pred_preprocessed_data['key'] = pred_preprocessed_data['order_number'].astype(int).astype(str) + \
                                            '-' + pred_preprocessed_data['operation_key'].astype(int).astype(str)
            pred_preprocessed_data.drop(columns=['forecast_man_hours'], inplace=True)
            pred_preprocessed_data = pred_preprocessed_data.merge(op_data_pred[['key','forecast_man_hours']], how='left', on='key')
            sap_preds = list(pred_preprocessed_data['forecast_man_hours'])
        else:
            sap_preds = list(pred_preprocessed_data['forecast_man_hours'])

        # Adjust predictions based on pred_threshold
        y_preds_adj = [sap_preds[i] + (y_preds[i][0] - sap_preds[i]) * pred_threshold for i in range(len(y_preds))]

        final_output = pd.DataFrame({
            'order_number': pred_preprocessed_data['order_number'],
            'operation_key': pred_preprocessed_data['operation_key'],
            'SAP_pred': sap_preds,
            'non_adjusted_pred': [round(pred[0], 1) for pred in y_preds],
            'adjusted_pred': [round(pred, 1) for pred in y_preds_adj]
        })

        return final_output

    def shap_plot(self):
        """Create and store a plot for feature importances using SHAP values.

        Returns:
            Plot
        """
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer.shap_values(self.X_test)

        shap.summary_plot(self.shap_values, self.X_test, show=False)
        return plt
