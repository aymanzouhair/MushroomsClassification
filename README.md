# Mushroom Classification

This project aims to apply 6 different classification methods on datasets to compare them. We will explore, analyze, and model data to classify mushrooms into two categories based on their characteristics.

The workflow includes data preprocessing, training different classifiers, evaluating results using classification reports, confusion matrices, precision metrics, and cross-validation.

## Project Structure

src/ classifiers/ : Contains the implementations of the classification models. utils/ : Contains utility functions, such as data preprocessing and evaluation. observation_notebooks/ : Contains Jupyter notebooks with trial-and-error analysis.

data/ : Contains the mushroom dataset from Kaggle

main.ipynb : The main file to run, contains all the instructions to carry out the tasks.


## Usage

To run the project, simply navigate to its root directory and launch the notebook in a Jupyter environment. Make sure you have installed the packages listed in `requirements.txt`.

## Libraries Used

- **NumPy** and **Pandas**: For data handling and manipulation.
- **Scikit-learn**: For model training, cross-validation, and performance evaluation.
- **Matplotlib** and **Seaborn**: For visualizing results, including confusion matrices and bar charts.
- **SciPy**: For calculating z-scores and detecting outliers.

## Data Preprocessing

The preprocessing steps include:

1. Loading data from the CSV file.
2. Removing outliers using z-scores to ensure clean data.
3. Normalizing features with `StandardScaler`.
4. Splitting the dataset into training and testing sets.

## Training and Evaluating Models

Each model is defined in its own class file under the `classifiers/` directory. The common structure of our classes is detailed in the report. For each model, the program starts by defining an instance of that class to use the model. Then, optimal hyperparameters are determined through functions within the classes (e.g., using grid search). The model is trained on our training set to be evaluated. A simple evaluation is performed first to provide a classification report and confusion matrix using our test set. Finally, cross-validation is done to estimate the precision of each model.

## Evaluation Metrics

### Classification Report

The classification report includes key metrics such as:

- **Precision**: How many selected elements are relevant.
- **Recall**: How many relevant elements are selected.
- **F1-Score**: The harmonic mean of precision and recall.
- **Support**: The number of occurrences of each class.

### Confusion Matrix

The confusion matrix shows the number of true positives, true negatives, false positives, and false negatives. It provides a visual evaluation of the classification model's performance.

### Cross-validation

A K-Fold cross-validation is performed on each model to evaluate them. The data is split into 5 subsets, and each model is trained and evaluated multiple times to get an accurate performance estimate. A graph is generated showing the average accuracy of each model to visualize their effectiveness.

## Conclusion

This project evaluates several chosen classification models on our mushroom dataset. By applying different classifiers and using cross-validation, we were able to compare their performance.
