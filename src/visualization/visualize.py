import matplotlib.pyplot as plt
from sklearn import tree
import pickle

def plot_decision_tree():
    """Plot the decision tree."""
    # Load the trained model
    with open('src/models/decision_tree_model.pkl', 'rb') as f:
        dtmodel = pickle.load(f)
    
    # Get feature names
    feature_names = dtmodel.feature_names_in_.tolist()  # Convert to list
    
    # Plot the tree
    plt.figure(figsize=(20,10))
    tree.plot_tree(dtmodel, feature_names=feature_names, filled=True, fontsize=10)
    plt.savefig('tree.png', dpi=300)
    plt.show()