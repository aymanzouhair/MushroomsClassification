from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 criterion=None, 
                 max_depth=None, 
                 min_samples_splits=None, 
                 min_samples_leaves=None, 
                 min_weight_fraction_leaves=None, 
                 random_state=42,
                 verbose=0):
        
        # Hyperparameters with defaults
        self.criteria = criterion if criterion else ["gini", "entropy"]
        self.max_depths = max_depth if max_depth else [None, 5, 10, 15, 20]
        self.min_samples_splits = min_samples_splits if min_samples_splits else [2, 5, 10, 15, 20]
        self.min_samples_leaves = min_samples_leaves if min_samples_leaves else [1, 2, 4, 6, 10]
        self.min_weight_fraction_leaves = min_weight_fraction_leaves if min_weight_fraction_leaves else [0, 0.1, 0.2]
        self.random_state = random_state
        self.verbose = verbose

        self.best_model = None
        self.best_params = None
        self.best_accuracy = 0

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):

        # Loop through all combinations of hyperparameters
        for criterion in self.criteria:
            for max_depth in self.max_depths:
                for min_samples_split in self.min_samples_splits:
                    for min_samples_leaf in self.min_samples_leaves:
                        for min_weight_fraction_leaf in self.min_weight_fraction_leaves:
                            model = DecisionTreeClassifier(
                                criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                random_state=self.random_state
                            )
                            model.fit(X_train, y_train)

                            # Evaluate the model's performance
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)

                            if self.verbose:
                                print(f"Criterion: {criterion}, Max Depth: {max_depth}, "
                                    f"Min Samples Split: {min_samples_split}, Min Samples Leaf: {min_samples_leaf}, "
                                    f"Min Weight Fraction Leaf: {min_weight_fraction_leaf}, Accuracy: {accuracy}")

                            # Update the best model if this one performs better
                            if accuracy > self.best_accuracy:
                                self.best_accuracy = accuracy
                                self.best_model = model
                                self.best_params = {
                                    'criterion': criterion,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'min_weight_fraction_leaf': min_weight_fraction_leaf
                                }

        # Best hyperparameters and accuracy
        if self.verbose:
            print("\nBest hyperparameters found:")
            print(f"Criterion: {self.best_params['criterion']}, Max Depth: {self.best_params['max_depth']}, "
                f"Min Samples Split: {self.best_params['min_samples_split']}, "
                f"Min Samples Leaf: {self.best_params['min_samples_leaf']}, "
                f"Min Weight Fraction Leaf: {self.best_params['min_weight_fraction_leaf']}, "
                f"Accuracy: {self.best_accuracy}")
        
        return self.best_params, self.best_accuracy

    def fit(self, X, y):
        # Train the decision tree model
        self.best_model.fit(X, y)
        return self
    
    def predict(self, X):
        # Use the trained best model to make predictions
        return self.best_model.predict(X)

    def predict_proba(self, X):
        # Predict class probabilities using the best model
        return self.best_model.predict_proba(X)
