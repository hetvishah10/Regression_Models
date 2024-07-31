# make_dataset.py

import pandas as pd

def load_data(filepath):
    """Load data from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the data."""
    # For this example, we assume no additional preprocessing is needed
    return df

if __name__ == "__main__":
    df = load_data('final.csv')
    df = preprocess_data(df)
    df.to_csv('preprocessed_data.csv', index=False)  # Save the preprocessed data