import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# will have to add in more imports as we go along depending on what we use
###scikit-learn is what we are using

def preprocess_data(dataset):
    
    # where format our data to make sure everything matches for reading
    
    return dataset 

def evaluation_model(model, X_test, y_test): ## if have other parameters, add them in here
        
    # where we evaluate the model
    ## predictions
    ## accuracy
        
    return model 
    
def main():
    # ################ LOADING DATA SETS ################
    # reliability datasets - the col reliability_percentage = % of buses that are ON TIME (1 - % = late)
    reliability_553 = pd.read_csv('data/MBTA_Reliability_Bus553.csv')
    reliability = pd.read_csv('data/MBTA_Reliability.csv')
    # alerts - gives dates buses are receiving alerts (particularly for delays) + reason
    alerts = pd.read_csv('data/BUS_Service_Alerts.csv') 
    # gives weather (avg temp + precipitation) for boston area
    boston_weather = pd.read_csv('data/boston_weather.csv')

    ### add in extra lines for more data
    print(reliability.head())

    # ################ CALL PREPROCESSING FUNCTION ################
    
    # model from scikit-learn library
    model = GaussianNB() # Gaussian Naive Bayes
    
    #train (like train a dog) model that we choose (from scikit-learn library) specific to what we need
    model.fit(X_train, y_train) # where X_train is the training data and y_train is the target labels
    
    # ################ CALL EVALUATION FUNCTION ################
    
if __name__ == '__main__':
    main()

