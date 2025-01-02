import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Identify rows where any of the z-scores exceed the threshold
    z_scores = pd.DataFrame(stats.zscore(df), columns=df.columns)
    outliers = z_scores[(np.abs(z_scores) > 3).any(axis=1)]
    # Drop the identified rows containing outliers
    df = df.drop(outliers.index)

    # Select the features and the labels in our dataset
    X = df.loc[:, df.columns != "class"]  # Features
    y = df['class']  # Class labels

    # Normalize features
    scaler = StandardScaler() # Init the scaler
    X_scaled = scaler.fit_transform(X) # Calculate and normalize X datas

    # Divide datas, keeping a similar class distribution in training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=101, stratify=y)

    return X_train, X_test, y_train, y_test
