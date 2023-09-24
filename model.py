def predict(forecasting_params, transformed_pred_df, op_data_pred, fitted_model, prep_models_fitted, no_orders_flag):
    order_numbers = []
    operations = []
    for element in forecasting_params['order_number_operation_key']:
        substrings = element.split('-')
        order_numbers.append(int(substrings[0]))
        try:
            operations.append(int(substrings[1]))
        except:
            operations.append(0)

    base_df = pd.DataFrame({
        'order_number': order_numbers,
        'operation_key': operations,
        'SAP_pred': np.nan,
        'non_adjusted_pred': np.nan,
        'adjusted_pred': np.nan
    })

    if no_orders_flag:
        pred_results = base_df
    else:
        pred_results = fitted_model.predict(transformed_pred_df, op_data_pred, forecasting_params['pred_threshold'], prep_models_fitted)
        
        # Handle previously unseen values in the target variable
        pred_results['clock_hours'] = pred_results['clock_hours'].apply(lambda x: x if x in fitted_model.classes_ else 0)

        # Remove from the base_df the values we were able to predict on
        pred_results['order_op'] = pred_results['order_number'].astype(int).astype(str) + '_' + pred_results['operation_key'].astype(int).astype(str)
        base_df['order_op'] = base_df['order_number'].astype(str) + '_' + base_df['operation_key'].astype(str)
        base_df = base_df[~base_df['order_op'].isin(pred_results['order_op'].tolist())]

        pred_results = pd.concat([pred_results, base_df])

    final_predictions = _handle_errors(pred_results)
    print(final_predictions)

