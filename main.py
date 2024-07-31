from src.data.make_dataset import load_data, preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_linear_regression, train_decision_tree, train_random_forest
from src.models.predict_model import make_predictions
from sklearn.model_selection import train_test_split

def main():
    """Main script to run the entire pipeline."""
    # Data loading and preprocessing
    df = load_data('data/raw/final.csv')
    df = preprocess_data(df)
    df.to_csv('preprocessed_data.csv', index=False)
    
    # Feature engineering
    df = build_features(df)
    df.to_csv('features_data.csv', index=False)
    
    # Prepare data for training
    x = df.drop('price', axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234, stratify=df['beds'])
    
    # Model training
    lr_train_mae, lr_test_mae = train_linear_regression(x_train, y_train, x_test, y_test)
    dt_train_mae, dt_test_mae = train_decision_tree(x_train, y_train, x_test, y_test)
    rf_train_mae, rf_test_mae = train_random_forest(x_train, y_train, x_test, y_test)
    
    # Print MAE values
    print(f"Linear Regression Train MAE: {lr_train_mae}")
    print(f"Linear Regression Test MAE: {lr_test_mae}")
    print(f"Decision Tree Train MAE: {dt_train_mae}")
    print(f"Decision Tree Test MAE: {dt_test_mae}")
    print(f"Random Forest Train MAE: {rf_train_mae}")
    print(f"Random Forest Test MAE: {rf_test_mae}")
    
    # Make predictions (without redundant MAE calculations)
    make_predictions(x_test, y_test)

if __name__ == "__main__":
    main()