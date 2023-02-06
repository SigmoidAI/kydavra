'''
Created with love by Sigmoid
@Author - Basoc Nicoleta-Nina - nicoleta.basoc28@gmail.com
'''
import pandas as pd
import numpy as np
from .errors import MissingDataError, NonNumericDataError, NoSuchColumnError


class PhikSelector:

    def __init__(self, min_corr = 0.3, max_corr = 0.8, erase_corr = True) -> None:
        """
            Setting up the model
        :param min_corr: float, between 0 and 1, default = 0.5
            The minimal positive correlation value that must the feature have with y_column
        :param max_corr: float, between 0 and 1, default = 0.8
            The maximal positive correlation value that must the feature have with y_column
        :param erase_corr: boolean, default = False
            If set as False the selector doesn't erase features that are highly correlated between themselves
            If set as True the selector does erase features that are highly correlated between themselves
        """

        self.min_corr = min_corr
        self.max_corr = max_corr
        self.erase_corr = erase_corr
    
    def index_to_cols(self, index_list : int) -> list:
        '''
            Converting the indexes list into names of features that are included in model
        :param index_list: list
            A list of Indexes that should be converted into a list of feature names
        :return: list
            A list with feature names
        '''
        return [self.X_columns[i] for i in index_list]
        
    def select(self, dataframe : pd.DataFrame, target : str) -> list:
        """
            Selecting the most important columns
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param target: str
             The column name of the value that we what to predict
        :return: list
            The list of features that our algorithm selects as the best ones
        """
        
        # Verifying if data frame has NaN-values.
        if dataframe.isna().values.any():
            raise MissingDataError("The passed data frame contains missing values")
        
        # If the given target value doesn't match any column in the DataFrame raise an error
        if target not in dataframe:
            raise NoSuchColumnError(f'No such column {target}')
        
        # If a column has string values raise an error.
        for col in dataframe:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                raise NonNumericDataError('The given DataFrame contains non-numeric values')
                
        # Getting the list with names of columns without the target one.
        self.X_columns = [col for col in dataframe.columns if col != target]
        
        # Getting the feature columns.
        X = dataframe.drop([target], axis=1).columns
        
        # Creating an empty list for correlated columns.
        correlated_indexes = []
        
        # Initialising the phik correlation on the dataframe   . 
        dataframe.corr()
        
        # Getting the target-column index.
        target_index = list(dataframe.columns).index(target)
        # Getting the correlation matrix.
        phik_corr_matrix= dataframe.phik_matrix(interval_cols=X)
        
        # Generating the correlation matrix
        self.corr_table = dataframe.phik_matrix(interval_cols=X)
        corr_matrix = self.corr_table.values
    
        # Searching for the columns correlated with the target-column.
        for i in range(len(corr_matrix[target_index])):
            if abs(corr_matrix[target_index][i]) > self.min_corr and abs(corr_matrix[target_index][i]) < self.max_corr:
                correlated_indexes.append(i)
                
        # Creating a list with the names of columns correlated with the y-column.
        self.correlated_cols = self.index_to_cols(correlated_indexes)
        
        # If we chose to erase a column from the correlated pairs we erase one of them.
        if self.erase_corr:
            cols_to_remove = []
            for i in correlated_indexes:
                for j in correlated_indexes:
                    if abs(corr_matrix[i][j]) > self.max_corr and abs(corr_matrix[i][j]) < self.max_corr:
                        cols_to_remove.append(self.X_columns[i])
            cols_to_remove = set(cols_to_remove)
            # Removing the chosen columns.
            for col in cols_to_remove:
                self.correlated_cols.remove(col)
                
        return self.correlated_cols
