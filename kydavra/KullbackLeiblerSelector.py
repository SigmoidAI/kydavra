"""
Created with love by Sigmoid
@Author - Smocvin Denis - denissmocvin@gmail.com
"""
from scipy.special import rel_entr
import numpy as np
import pandas as pd
from .errors import MissingDataError, NonNumericDataError, NoSuchColumnError


class KullbackLeiblerSelector:
    def __init__(self, EPS: float = 0.0001, min_divergence: int = 0, max_divergence: int = 1):
        """
        Setting up the algorithm.
        :param EPS:
            A small value to add to feature columns' probabilities in order to avoid division by 0
        :param min_divergence:
            The minimum divergence that the columns should have
        :param max_divergence:
            The maximum divergence that the columns should have
        """
        self.EPS = EPS
        self.max_divergence = max_divergence
        self.min_divergence = min_divergence

    def __calculate_divergence(self, q: 'np.array', p: 'np.array') -> float:
        """
        Calculating the Kullback-Leibler divergence between a feature array and a target array.
        :param q: numpy array
            Feature array
        :param p: numpy array
            Target array
        :return: float
            The calculated Kullback-Leibler divergence
        """
        q = (q / sum(q)) + self.EPS
        p = p / sum(p)

        # calculating relative entropy which is the same thing as KL divergence
        vec = rel_entr(p, q)
        vec = np.ma.masked_invalid(vec).compressed()

        return np.sum(vec)

    def select(self, dataframe: 'pd.Dataframe', target: str) -> list:
        """
        Selecting columns according to their divergence.
        :param dataframe: pandas DataFrame
            The DataFrame on which the selection is applied
        :param target: string
            The name of the target column
        :return: list
            The list of columns that have a divergence relative to the target column between min_divergence and max_divergence
        """

        self.dataframe = dataframe
        self.X_columns = [col for col in self.dataframe.columns if col != target]

        # if the given y_column doesn't match any column in the DataFrame raise an error
        if target not in self.dataframe:
            raise NoSuchColumnError(f'No such column {target}')

        # if a column has string values raise an error
        for col in self.dataframe:
            if not pd.api.types.is_numeric_dtype(self.dataframe[col]):
                raise NonNumericDataError('The given DataFrame contains non-numeric values !')

        if self.dataframe.isna().values.any():
            raise MissingDataError('The given DataFrame contains NaN values !')

        # defining a list to store the names of selected columns
        selected_cols = []

        for col in self.X_columns:
            # calculating the Kullback Leibler divergence of the column
            d = self.__calculate_divergence(self.dataframe[col].values, self.dataframe[target].values)

            # selecting the column if its divergence is between min_divergence and max_divergence
            if self.min_divergence < d < self.max_divergence:
                selected_cols.append(col)

        return selected_cols