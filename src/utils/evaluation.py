from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Perform K-Fold cross-validation on the given model
def cross_validate_kfold(model, X_train, y_train, n_splits=5):
    print(f"\nStarting {n_splits}-Fold cross-validation...")

    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    # Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # Split data into training and validation sets for this fold
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train the model on the current fold
        model.fit(X_train_fold, y_train_fold)

        # Predict on the validation set
        y_pred = model.predict(X_val_fold)

        # Calculate accuracy for this fold
        fold_accuracy = accuracy_score(y_val_fold, y_pred)
        fold_accuracies.append(fold_accuracy)

        print(f"Fold {fold + 1}: Accuracy = {fold_accuracy:.4f}")

    # Calculate mean accuracy across all folds
    mean_accuracy = np.mean(fold_accuracies)
    print(f"\nCross-Validation completed.\nMean Accuracy: {mean_accuracy:.4f}")

    return mean_accuracy


# Evaluate the trained model on the test set
def evaluate(model, X_test, y_test):

    print("\nEvaluating model on the test set...")
    y_pred = model.predict(X_test)

    # Print classification report including precision, recall, and F1-score
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Visualize the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)  # Calculate the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")

    # Extract the model's class name for the title
    model_name = type(model).__name__
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')  # Add model name in the title
    plt.show()

# Plot the best accuracies for all methods
def plot_all_accuracies(models, accuracies):

    # Build the figure
    plt.figure(figsize=(8, 6))

    # Create a horizontal bar plot
    sns.barplot(x=accuracies, y=models, palette="Blues_r", hue=models)

    # Add annotations on the points
    for i, accuracy in enumerate(accuracies):
        plt.text(accuracy + 0.01, i, f"{accuracy * 100:.1f}%", va='center')  # Accuracy

    # Add title and labels
    plt.xlabel("Accuracy")
    plt.ylabel("Models")
    plt.title("Comparison of Model Accuracies")
    plt.xlim(0, max(accuracies) + 0.05)
    plt.tight_layout()
    plt.show()