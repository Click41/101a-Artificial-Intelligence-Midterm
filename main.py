import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# will have to add in more imports as we go along depending on what we use
###scikit-learn is what we are using

def preprocess_data(dataset):
    
    ############where format our data to make sure everything matches for reading
    ### can change according to what we need
    
     # Assuming dataset has features (X) and target variable (y)
    X = dataset.drop('target_column', axis=1)
    y = dataset['target_column']
    ## i think correlates to X_train, X_test, y_train, y_test

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(X)

    # Assuming you want to concatenate the preprocessed features and target variable
    dataset_preprocessed = pd.concat([pd.DataFrame(X_preprocessed), y], axis=1)
    
    return dataset_preprocessed
    

def evaluation_model(model, X_test, y_test): ## if have other parameters, add them in here
        
    # where we evaluate the model
    ## predictions
    ## accuracy
        
    return model 
    
def main():
    # ################ LOADING DATA SETS ################
    # reliability datasets - the col reliability_percentage = % of buses that are ON TIME (1 - % = late)
    # unreliable_percentage = % of times/buses that are late
    reliability_553 = pd.read_csv('./data/MBTA_Bus_Reliability_Bus553.csv')
    reliability = pd.read_csv('./data/MBTA_Bus_Reliability.csv')
    # alerts - gives dates buses are receiving alerts (particularly for delays) + reason
    alerts = pd.read_csv('data/BUS_Service_Alerts.csv') 
    alerts = pd.read_csv('data/BUS_Service_Alerts.csv')
    # gives weather (avg temp + precipitation) for boston area
    boston_weather = pd.read_csv('data/boston_weather.csv')
    

    ## combines datasets into one dataframe
    combined_data = pd.concat([reliability_553, reliability, alerts, boston_weather], axis=1)
    print(combined_data.columns)
    ## can takeout if we decide to use all 4 separately, just make sure to change function

    ### add in extra lines for more data
    print(combined_data.head())
    # print(reliability.head())

    # ################ CALL PREPROCESSING FUNCTION ################
    processed_data = preprocess_data(combined_data)

    ## split data into training-testing sets
    X = processed_data.drop('add_reliability_bool_col', axis=1)
    y = processed_data['add_reliability_bool_col']

    #split data into train-test 80-20
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

    # model from scikit-learn library
    model = GaussianNB() # Gaussian Naive Bayes
    

    #train (like train a dog) model that we choose (from scikit-learn library) specific to what we need
    model.fit(X_train, y_train) # where X_train is the training data and y_train is the target labels


    #Make the prediction
    y_pred_gnb = model.predict(X_test)

    #Use confusion matrix and classification report to check the model's performance
    conf_matrix = confusion_matrix(y_test, y_pred_gnb)

    # Display the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # ################ CALL EVALUATION FUNCTION ################
    evaluation_model(model, X_test, y_test)
    ## change parameters if changed in test Split/other places
    
if __name__ == '__main__':
    main()