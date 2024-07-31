import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_linear_regression(x_train, y_train, x_test, y_test):
    """Train a linear regression model and save it to a file."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Save the model
    with open('src/models/linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Calculate MAE for training and test sets
    train_mae = mean_absolute_error(y_train, model.predict(x_train))
    test_mae = mean_absolute_error(y_test, model.predict(x_test))
    
    return train_mae, test_mae

def train_decision_tree(x_train, y_train, x_test, y_test):
    """Train a decision tree model and save it to a file."""
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    
    # Save the model
    with open('src/models/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Calculate MAE for training and test sets
    train_mae = mean_absolute_error(y_train, model.predict(x_train))
    test_mae = mean_absolute_error(y_test, model.predict(x_test))
    
    return train_mae, test_mae

def train_random_forest(x_train, y_train, x_test, y_test):
    """Train a random forest model and save it to a file."""
    model = RandomForestRegressor(n_estimators=200, criterion='absolute_error')
    model.fit(x_train, y_train)
    
    # Save the model
    with open('src/models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Calculate MAE for training and test sets
    train_mae = mean_absolute_error(y_train, model.predict(x_train))
    test_mae = mean_absolute_error(y_test, model.predict(x_test))
    
    return train_mae, test_mae

def load_trained_model(model_path):
    """Load a trained model from a file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model