from sklearn.model_selection import train_test_split
import pandas as pd


def TestTrainSplit(file_name, features, label, ratio):

    dataset = pd.read_csv(file_name)
    X = dataset[features].values
    y = dataset[label].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state = 0)

    return X_train, X_test, y_train, y_test
