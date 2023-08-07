from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os
import io
from joblib import dump
import streamlit as st
import numpy as np
import pandas as pd
from utils import *
import time


# db_raw_path = 'https://raw.githubusercontent.com/vcubo/beta_0.1/main/VCDB_230719v4_beta.csv'
# st.session_state.df_base = import_df(db_raw_path) # secondary dataframe for individual project operations


# Function to train a random forest model
st.cache_data()
def train_rf(X_train, y_train, n_estimators=100, random_state=42):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    X_train = preprocess_data(X_train)
    rf.fit(X_train, y_train)
    return rf

# Function to make predictions and compute RMSE
st.cache_data()
def predict_and_evaluate(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return y_pred, rmse

st.cache_data()
def preprocess_data(X_train):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    # X_test = imputer.transform(X_test)
    return X_train


# Function to convert models dict to bytes
st.cache_data()
def convert_models_to_bytes(models_dict):
    """
    Convert models dictionary to bytes using joblib and io.BytesIO
    :param models_dict: Dictionary of trained models to be converted to bytes
    :return: byte stream of the models dictionary
    """
    models_byte_stream = io.BytesIO()
    dump(models_dict, models_byte_stream)
    models_byte_stream.seek(0)  # Go back to the start of the stream
    return models_byte_stream



st.cache_data()
def two_steps_ml(df_base):
    """
    This function trains a two-step machine learning model using Random Forest Regression. In the first step, it predicts intermediate
    targets based on provided features. In the second step, it uses the original features and the predictions of the intermediate targets
    to predict the final targets. The function returns trained models and encoders.
    
    Parameters:
    df_base (pandas.DataFrame): The original dataframe.

    Returns:
    dict: A dictionary containing trained models and encoders for each target.
    """
    st.caption('Started training')
    start = time.time()
    df = df_base.copy(deep=True)
    # Define features and intermediate targets
    categorical_features = ['QUARTER', 'COUNTRY', 'LOB', 'SITE', 'PR_SIZE', 'MC_SIZE']
    intermediate_targets = ['COUNTRY_RMEAN', 'LOB_RMEAN', 'SITE_RMEAN', 'PSIZE_RMEAN', 'CSIZE_RMEAN',
                            'SOC', 'SOC_MIT', 'PROC', 'PROC_MIT', 'ENG', 'ENG_MIT', 'WEA', 'WEA_MIT', 
                            'MGM', 'MGM_MIT', 'SOC_EMEAN', 'PROC_EMEAN', 'ENG_EMEAN', 'WEA_EMEAN', 'MGM_EMEAN']
    final_targets = ['DEV_RAN', 'DEV_EVE', 'DEV_TOT']

    # Prepare the one-hot encoder
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(df[categorical_features])
    
    # Use the encoder to transform the categorical variables
    X = encoder.transform(df[categorical_features])
    
    # Convert it back to DataFrame
    X = pd.DataFrame(X, columns=encoder.get_feature_names(categorical_features))

    y = df[intermediate_targets + final_targets]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model for each intermediate target and store predictions
    intermediate_predictions_train = pd.DataFrame()
    intermediate_predictions_test = pd.DataFrame()
    models = {}
    
    for target in intermediate_targets:
        # Split data into training and test sets for each target
        X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.2, random_state=42)

        rf = train_rf(X_train, y_train)
        y_pred_train, rmse_ytr = predict_and_evaluate(rf, X_train, y_train)
        st.write(f"RMSE for intermediate target {target}: ", rmse_ytr)
        y_pred_test, rmse_yte = predict_and_evaluate(rf, X_test, y_test)
        
        # X_train[target] = y_pred_train
        # X_test[target] = y_pred_test
        intermediate_predictions_train.loc[:, target] = y_pred_train
        intermediate_predictions_test.loc[:, target] = y_pred_test
        models[target] = rf  # Save the model

    # At this point, X_train and X_test include original features and the predictions of the intermediate targets

    for target in final_targets:
        # Split data into training and test sets for each target
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y[target], test_size=0.2, random_state=42)

        rf = train_rf(X_train_final, y_train_final)
        _, rmse = predict_and_evaluate(rf, X_test_final, y_test_final)
        st.write(f"RMSE for final target {target}: ", rmse)
        models[target] = rf  # Save the model
    
    st.caption(f'Completed training in **{np.round(time.time()-start,2)} seconds**')
    return [models, encoder]

st.cache_data()
def predict_features(feature_values, rf_models, enc):
    # Create a dataframe from the provided feature values (reshaped to 1 row and n columns)
    X_new = pd.DataFrame(np.array(feature_values).reshape(1, -1), columns=['QUARTER', 'COUNTRY', 'LOB', 'SITE', 'PR_SIZE', 'MC_SIZE'])

    # One-hot encode the dataframe using the same encoder instance
    X_new_encoded = enc.transform(X_new)
    st.dataframe(X_new_encoded)
    # # The encoded data needs to be converted to a DataFrame since the encoder returns a numpy array
    # X_new_encoded_df = pd.DataFrame(X_new_encoded)

    # Initialize empty dictionary to store predicted values
    results = {}

    # Iterate over the random forest models
    for target, rf in rf_models.items():
        # Predict the target
        prediction = rf.predict(X_new_encoded)
        # Store the prediction in the results dictionary
        results[target] = prediction[0]

    return results

# # Create trained models and convert models_dict to byte stream
# if 'models' not in st.session_state: 
#     st.session_state.models = two_steps_ml(st.session_state.df_base)
#     st.session_state.model_byte_stream = convert_models_to_bytes(st.session_state.models)


# # Download models using Streamlit's download button
# st.download_button(
#     label="Download trained models",
#     data=st.session_state.model_byte_stream,
#     file_name='trained_models.joblib',
#     mime='application/octet-stream'
# )

