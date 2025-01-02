from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class NeuralNetworkClassifier:
    def __init__(self, hidden_layer_sizes=None, learning_rates=None, max_iter=200): # We chose 200 as max_iter as it ensures a balance between convergence and training time
        
        # Define different hidden layer sizes that our model explores during training.
        self.hidden_layer_sizes = hidden_layer_sizes if hidden_layer_sizes else [(10,), (50,), (100,)] # One hidden layer with 10,50 or 100 units
        
        # Define different learning rates that our model explores during training.
        self.learning_rates = learning_rates if learning_rates else [0.001, 0.01, 0.1] # Default learning rates : 0.001, 0.01 and 0.1

        self.max_iter = max_iter
        self.best_model = None
        self.best_params = None
        self.best_accuracy = 0

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        
        # We train our model with different values of hidden layer sizes and learning rates to find the ones the give the best results
        for hidden_layer_size in self.hidden_layer_sizes: 
            for learning_rate in self.learning_rates:
                # Initialize and train the model
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_size, learning_rate_init=learning_rate,
                                       max_iter=self.max_iter, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate the performance
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                print(f"Hidden Layer Sizes: {hidden_layer_size}, Learning Rate: {learning_rate}, Accuracy: {accuracy}")

                # Update the best results
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = model
                    self.best_params = {'hidden_layer_sizes': hidden_layer_size, 'learning_rate': learning_rate}

        print("\nBest hyperparameters found:")
        print(f"Hidden Layer Sizes: {self.best_params['hidden_layer_sizes']}, Learning Rate: {self.best_params['learning_rate']}, Accuracy: {self.best_accuracy}")
        return self.best_params, self.best_accuracy
    
    def fit(self, X, y):
        #Train the neural networks model
        self.best_model.fit(X, y)
        return self


    def predict(self, X):
        # Use the best model to make predictions on new data
        return self.best_model.predict(X)