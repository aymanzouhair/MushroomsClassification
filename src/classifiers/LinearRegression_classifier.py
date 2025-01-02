from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score


class LinearRegressionClassifier:
    def __init__(self, alpha_values=None):
        # List of alpha values to test for Ridge regularization
        self.alpha_values = alpha_values if alpha_values else [0.01, 0.1, 1.0, 10.0, 100.0]
        self.best_model = None  # Best Ridge model found
        self.best_alpha = None  # Best alpha value
        self.best_accuracy = 0  # Best accuracy achieved

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        # Search for the best hyperparameters (alpha)
        for alpha in self.alpha_values:
            # Initialize and train the Ridge model with the current alpha
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            # Predict results on the test set
            y_pred = model.predict(X_test)

            # Convert predictions to binary classes using a 0.5 threshold
            y_pred_class = (y_pred >= 0.5).astype(int)

            # Calculate the accuracy
            accuracy = accuracy_score(y_test, y_pred_class)

            print(f"Alpha: {alpha}, Accuracy: {accuracy}")

            # Update the best model if necessary
            if accuracy > self.best_accuracy:
                self.best_model = model
                self.best_alpha = alpha
                self.best_accuracy = accuracy

        print("\nBest hyperparameter found:")
        print(f"Alpha: {self.best_alpha}, Accuracy: {self.best_accuracy}")
        return self.best_alpha, self.best_accuracy

    def fit(self, X, y):
        # Train the decision tree model
        self.best_model.fit(X, y)
        return self

    def predict(self, X):
        # Predict using the best model
        y_pred = self.best_model.predict(X)
        return (y_pred >= 0.5).astype(int)
