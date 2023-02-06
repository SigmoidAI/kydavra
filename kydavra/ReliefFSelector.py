'''
Created with love by Sigmoid
@Author - iurie.cius.personal@gmail.com
'''

# Importing all needed libraries.
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .errors import MissingDataError, NonNumericDataError, NoSuchColumnError


class ReliefFSelector:
    def __init__(self, n_neighbors: int = 5, n_features: int = 10) -> None:
        """
            Setting up the algorithm
        :param n_neighbors: int, default = 5
            The number of neighbors to consider when assigning feature
            importance scores. If a float number is provided, that percentage of
            training samples is used as the number of neighbors.
            More neighbors results in more accurate scores, but takes longer.
        :param n_features: int, default = 10
            The number of top features (according to the relieff score) to
            retain after feature selection is applied.
        """
        self.n_neighbors = n_neighbors
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

        # Identify neighbors.
        nbrs = NearestNeighbors(n_neighbors=n_samples,
                                algorithm='kd_tree').fit(X)

        # Get the number of classes.
        c = np.unique(y).tolist()

        # Compute the distances and indices of the neighbors.
        distances, indices = nbrs.kneighbors(X)

        # Iterating through every sample in the dataset.
        for i in range(n_samples):

            # Initialize the near miss and near hit data structures.
            near_miss = dict()
            near_hit = []

            # Getting all the classes of the target column.
            for label in c:
                near_miss[label] = []

            # Identify k nearest hits and k nearest misses (using distance array).
            for n in range(1, self.n_neighbors):
                if y[i] == y[indices[i][n]]:
                    near_hit.append(indices[i][n])
                else:
                    near_miss[y[indices[i][n]]].append(indices[i][n])

            # Initialize the dictionary for storing the probabilities of every class.
            p_dict = dict()
            p_label_index = float(len(y[y == y[i]])) / float(n_samples)

            # Computing the probabilities of every class.
            for label in c:
                p_label_c = float(len(y[y == label])) / float(n_samples)
                p_dict[label] = p_label_c / (1 - p_label_index)

            # Initialize near hit terms.
            near_hit_term = np.zeros(n_features)

            # Compute near hit terms.
            for element in near_hit:
                near_hit_term = np.array(
                    near_hit_term) + np.array(abs(X[i, :] - X[element, :]))

            # Initialize near miss terms.
            near_miss_term = dict()

            # Compute near miss terms.
            for (label, miss_list) in near_miss.items():
                near_miss_term[label] = np.zeros(n_features)
                for element in miss_list:
                    near_miss_term[label] = np.array(abs(
                        X[i, :] - X[element, :])) + np.array(near_miss_term[label])

                # Update the Score.
                score += near_miss_term[label] / (self.n_neighbors * n_samples)

            # Update the Score.
            score -= near_hit_term / self.n_neighbors

        # Sorting the indexes based on the score.
        res = np.argsort(score, 0)
        idx = res[::-1][:self.n_features]

        # Return the top n features.
        return list(dataframe.iloc[:, idx].columns)
