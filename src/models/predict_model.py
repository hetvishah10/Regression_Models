import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error

def load_trained_model(model_path):
    """Load a trained model from a file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions(x_test=None, y_test=None):
    if x_test is None or y_test is None:
        df = pd.read_csv('features_data.csv')
        x_test = df.drop('price', axis=1)
        y_test = df['price']
    
    lr_model = load_trained_model('src/models/linear_regression_model.pkl')
    dt_model = load_trained_model('src/models/decision_tree_model.pkl')
    rf_model = load_trained_model('src/models/random_forest_model.pkl')
    
    lr_predictions = lr_model.predict(x_test)
    dt_predictions = dt_model.predict(x_test)
    rf_predictions = rf_model.predict(x_test)
    
    lr_mae = mean_absolute_error(y_test, lr_predictions)
    dt_mae = mean_absolute_error(y_test, dt_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    
    print(f'Linear Regression Test MAE: {lr_mae}')
    print(f'Decision Tree Test MAE: {dt_mae}')
    print(f'Random Forest Test MAE: {rf_mae}')