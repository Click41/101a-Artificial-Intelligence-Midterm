import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def preprocess_data(dataset, test_size=0.2, random_state=42):
    
    # Assuming dataset has features (X) and target variable (y)
    columns_to_drop = ['gtfs_route_id', 'mode_type', 'peak_offpeak_ind']
    X = dataset.drop(columns=columns_to_drop, axis=1)  # Features (input)
    y = dataset['unreliable_percentage']  # Target variable (output)

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding for categorical variables
    ])

    # Applies the appropriate preprocessing steps to the respective types of features.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Preprocess the entire dataset
    X_preprocessed = preprocessor.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def evaluation_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # # Print or return the metrics
    # print(f"Mean Absolute Error: {mae}")
    # print(f"Mean Squared Error: {mse}")
    # print(f"Root Mean Squared Error: {rmse}")
    # print(f"R-squared: {r2}")

    return model
    
def main():
    # ################ LOADING DATA SETS ################
    # reliability datasets - the col reliability_percentage = % of buses that are ON TIME (1 - % = late)
    # unreliable_percentage = % of times/buses that are late
        
    # alerts - gives dates buses are receiving alerts (particularly for delays) + reason
    alerts = pd.read_csv('./data/BUS_Service_Alerts.csv') 
    
    # gives weather (avg temp + precipitation) for boston area
    boston_weather = pd.read_csv('./data/boston_weather.csv')
    
    # one that we use for most data accuracy
    reliability = pd.read_csv('./data/MBTA_Bus_Reliability.csv')

    # ################ CALL PREPROCESSING FUNCTION ################
    # Preprocesses the entire dataset
    X_train, X_test, y_train, y_test = preprocess_data(reliability)    
    
    # ################ MODEL TRAINING ################
    # model from scikit-learn library
    model = LinearRegression() # Linear Regression model

    #train (like train a dog) model that we choose (from scikit-learn library) specific to what we need
    model.fit(X_train, y_train) # where X_train is the training data and y_train is the target labels
    
    # ################ CALL EVALUATION FUNCTION ################
    evaluation_model(model, X_test, y_test)
    
if __name__ == '__main__':
    main()