from extract_features_helpers import get_features_noneighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def train_model(atoms, average_steptimes):
    """
    Given a list of atoms objects and a list of average steptimes, 
    train a StandardScaler and KNeighborsRegressor from sklearn.
    
    Parameters:
        atoms (ase.Atoms list): list of ase.Atoms objects.
        average_steptimes (float list): list of average time per step,
            usually given in coreseconds. Values should correspond to the
            atoms list by index.
            
    Returns:
        model ((scaler, regressor) tuple): a tuple that contains the trained scaler and regressor.
    """
    
    features = list(map(get_features_noneighbors, atoms))
    
    scaler = StandardScaler()
    scaler.fit(features)
    X = scaler.transform(features)
    
    regressor =  KNeighborsRegressor(weights="uniform", algorithm="auto", n_jobs=4, leaf_size=30)
    regressor.fit(X, average_steptimes)
    
    return (scaler, regressor)

def make_prediction_function(model):
    """
    Given a model (scaler, regressor), create a function of type 
    ase.Atoms -> average_steptime which uses the model to predict
    the average time per step of running DFT on given atoms object.
    
    Parameters:
        model ((scaler, regressor) tuple): a tuple that contains the trained scaler and regressor.
        
    Returns:
        prediction_function(ase.Atoms -> average_steptime function):
            a function that predicts the average steptime of given initial atoms object.
    """
    
    scaler, regressor = model
    
    # making prediction function from given model
    def prediction_function(atoms):
        features = get_features_noneighbors(atoms)
        x = scaler.transform([features])
        y_predict = regressor.predict(x)
        return y_predict[0]
    
    return prediction_function