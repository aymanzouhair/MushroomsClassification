from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


class PerceptronClassifier:
    def __init__(self, eta0_values=None, alpha_values=None, max_iter=100000, tol=1e-3):
        # Hyperparameter values to search for eta0 and alpha
        self.eta0_values = eta0_values if eta0_values else [0.0001, 0.001, 0.01, 0.1, 1.0]
        self.alpha_values = alpha_values if alpha_values else [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
        self.max_iter = max_iter
        self.tol = tol  # Add tolerance to control convergence

        # For storing the best model and parameters
        self.best_model = None
        self.best_params = None
        self.best_accuracy = 0

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        # Loop through all combinations of eta0 and alpha
        for eta0 in self.eta0_values:
            for alpha in self.alpha_values:
                # Create and train the Perceptron model with the current eta0 and alpha
                model = Perceptron(max_iter=self.max_iter, eta0=eta0, alpha=alpha, tol=self.tol)
                model.fit(X_train, y_train)

                # Predict labels on the test set
                y_pred = model.predict(X_test)

                # Calculate the accuracy on the test data
                accuracy = accuracy_score(y_test, y_pred)

                print(f"eta0: {eta0}, Alpha: {alpha}, Accuracy: {accuracy}")

                # Update the best model if the current model performs better
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = model
                    self.best_params = {'eta0': eta0, 'alpha': alpha}

        print(f"\nBest hyperparameters found:")
        print(f"eta0: {self.best_params['eta0']}, Alpha: {self.best_params['alpha']}, Accuracy: {self.best_accuracy}")
        return self.best_params, self.best_accuracy

    def fit(self, X, y):
        # Train the decision tree model
        self.best_model.fit(X, y)
        return self

    def predict(self, X):
        # Use the best model to make predictions on new data
        return self.best_model.predict(X)
