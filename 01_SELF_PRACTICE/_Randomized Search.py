import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# 1. Load sample dataset
data = load_iris()
X, y = data.data, data.target

# 2. Define the model
clf = RandomForestClassifier()

# 3. Define the continuous distributions (NOT a rigid grid!)
param_distributions = {
    'n_estimators': np.arange(10, 200, 10),           # Discrete choices
    'max_depth': [None, 3, 5, 10, 20],                # Discrete choices
    'min_samples_split': np.arange(2, 11),            # Range of integers
    'criterion': ['gini', 'entropy']                  # Categorical choices
}

# 4. Set up the Random Search: Here n_iter=20 means it will pick exactly 20 random combinations to test.
random_search = RandomizedSearchCV(
    estimator=clf, 
    param_distributions=param_distributions, 
    n_iter=20, 
    cv=5,                             # 5-Fold Cross Validation
    random_state=42, 
    n_jobs=-1                         # Use all available CPU cores
)

# 5. Run the search
random_search.fit(X, y)

# 6. Output the results
print(f"Best Score: {random_search.best_score_:.4f}")
print("Best Hyperparameters Found:")
print(random_search.best_params_)
