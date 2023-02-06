'''
Created with love by Sigmoid
@Author - iurie.cius.personal@gmail.com
'''

# Importing all needed libraries.
import numpy as np
import pandas as pd
from .errors import MissingDataError, NonNumericDataError, NoSuchColumnError


class FisherSelector:
    def __init__(self, n_features: int = 5) -> None:
        """
            Setting up the algorithm
        :param n_features: int, default = 5
            The number of top features (according to the relieff score) to
            retain after feature selection is applied.
        """
        self.n_features = n_features

    def select(self, dataframe: 'pd.DataFrame', target: str) -> list:
        """
            Selecting the most important columns.
        :param dataframe: pandas DataFrame
            Data Frame on which the algorithm is applied.
        :param target: str
            The column name of the value that we what to predict.
        :return: list
            The list of features that are selected by the algorithm as the best one.
        """

        if dataframe.isna().values.any():
            raise MissingDataError(
                "The passed data frame contains missing values.")

        # if a column has string values raise an error
        for col in dataframe:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                raise NonNumericDataError(
                    'The given DataFrame contains non-numeric values !')

        if target not in dataframe:
            raise NoSuchColumnError("The passed target column doesn't exist.")

        # Converting the data frame in numpy arrays.
        X = dataframe.drop([target], axis=1).values
        y = dataframe[target].values

        # Get the dimensionality of the DataFrame.
        n_samples, n_features = X.shape

        # Initialize the scores to 0.
        score = np.zeros(n_features)

        # Get the number of classes.
        classes = np.unique(y).tolist()

        # Initialize all the variables to 0.
        n_k, mean_k, mean_dataframe, std_dataframe = 0, 0, 0, 0

        # Iterating through every feature in the dataset.
        for i in range(n_features):

            # Iterating through every class in the dataset.
            for c in classes:
                # Compute the size of the k-th class.
                n_k = dataframe[target].value_counts().get(c)

                # Compute the mean of the k-th class.
                mean_k = dataframe.groupby(target).mean()[
                    dataframe.columns[i]].get(c)

                # Compute the mean of the whole dataset.
                mean_dataframe = dataframe[dataframe.columns[i]].mean()

                # Compute the standard deviation of the whole dataset.
                std_dataframe = dataframe[dataframe.columns[i]].std()

                # Update the score.
                score[i] += (n_k * (mean_k - std_dataframe)**2)

            # Update the score
            score[i] /= mean_dataframe**2

        # Sorting the indexes based on the score.
        res = np.argsort(score, 0)
        idx = res[::-1][:self.n_features]

        # Return the top n features.
        return list(dataframe.iloc[:, idx].columns)
