from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVMClassifier:
    def __init__(self, kernels=None, C_values=None):
        
        # Define different kernels our model explores during training.
        self.kernels = kernels if kernels else ['linear', 'poly', 'rbf', 'sigmoid']
        
        # Define different values of our regularization parameter C that our model explores during training.
        self.C_values = C_values if C_values else [0.001, 0.01, 0.1, 1, 2]

        self.best_model = None
        self.best_params = None
        self.best_accuracy = 0

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
       
        for kernel in self.kernels:
            for C in self.C_values:
                # Initialize and train the model
                model = SVC(kernel=kernel, C=C)
                model.fit(X_train, y_train)

                # Evaluate the performance
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                print(f"Kernel: {kernel}, C: {C}, Accuracy: {accuracy}")

                # Update the best results
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = model
                    self.best_params = {'kernel': kernel, 'C': C}

        print("\nBest hyperparameters found:")
        print(f"Kernel: {self.best_params['kernel']}, C: {self.best_params['C']}, Accuracy: {self.best_accuracy}")
        return self.best_params, self.best_accuracy
    
    def fit(self, X, y):
        #Train the SVM model
        self.best_model.fit(X, y)
        return self

    def predict(self, X):
        # Use the best model to make predictions on new data
        return self.best_model.predict(X)