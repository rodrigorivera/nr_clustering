import pandas as pd
import logging
from sklearn import preprocessing


def generate_smape_table(df: pd.DataFrame
                         ) -> pd.DataFrame:
    df_predictions = df
    model_length = df_predictions['model'].unique()
    flag_length = df_predictions['future_flag'].unique()
    df_temp = pd.DataFrame()

    for future_flag, model in [(future_flag, model) for future_flag in flag_length for model in model_length]:
        condition = (df_predictions['future_flag'] == future_flag) & (df_predictions['model'] == model)
        # Create x, where x the 'scores' column's values as floats
        x = df_predictions[condition]['SMAPE'].values.reshape(-1, 1)
        # .values.astype(float)]

        # Create a minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()

        # Create an object to transform the data to fit minmax processor
        x_scaled = min_max_scaler.fit_transform(x)

        # Run the normalizer on the dataframe
        # RESULTS['scaled_SMAPE'] = x_scaled
        df_normalized = pd.DataFrame(x_scaled, columns=['SMAPE'])
        df_normalized['future_flag'] = future_flag
        df_normalized['model'] = model
        df_normalized['predicted'] = df_predictions['predicted']
        df_normalized['target'] = df_predictions['target']
        df_normalized['column'] = df_predictions['column']
        df_normalized['dataset'] = df_predictions['dataset']
        df_temp = pd.concat([df_temp, df_normalized])

        logging.debug('generate_smape_table:')
        logging.debug(df_temp)

    return df_temp