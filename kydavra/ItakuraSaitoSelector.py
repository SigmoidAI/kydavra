"""
Created with love by Sigmoid
@Author - Smocvin Denis - denissmocvin@gmail.com
"""

import numpy as np
import pandas as pd
import math
from .errors import MissingDataError, NonNumericDataError, NoSuchColumnError


class ItakuraSaitoSelector:
    def __init__(self, EPS: float = 0.0001, min_divergence: int = 0, max_divergence: int = 10):
        """
        Initialize the algorithm.
        :param EPS:
            A small number to replace 0 probability in the distributions in order to avoid division by 0
        :param min_divergence:
            The minimum divergence that the columns should have
        :param max_divergence:
            The maximum divergence that the columns should have
        """

        self.EPS = EPS
        self.min_divergence = min_divergence
        self.max_divergence = max_divergence

    def __calculate_divergence(self, q: 'np.array', p: 'np.array') -> float:
        d = 0

        # calculating the distributions
        p = p / sum(p)
        q = q / sum(q)

        p = [self.EPS if x == 0 else x for x in p]
        q = [self.EPS if x == 0 else x for x in q]

        # calculating the divergence
        for i in range(len(p)):
            d += (p[i] / q[i]) - math.log(p[i] / q[i]) - 1

        return d / len(p)

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

        # raise an error if there are NaN values in the DataFrame
        if self.dataframe.isna().values.any():
            raise MissingDataError('The given DataFrame contains NaN values !')

        # defining a list to store the names of selected columns
        selected_cols = []

        for col in self.X_columns:
            # calculating the Itakura-Saito divergence of the column
            d = self.__calculate_divergence(self.dataframe[col], self.dataframe[target])

            # selecting the column if its divergence is between min_divergence and max_divergence
            if self.min_divergence < d < self.max_divergence:
                selected_cols.append(col)

        return selected_cols