'''
Created with love by Sigmoid.
@Author - Spînu Lavinia - laviniaspinu01@gmail.com
'''
# Import all needed libraries.
import pandas as pd
from sklearn.decomposition import FastICA
from .errors import MissingDataError, NonNumericDataError, NoSuchColumnError, DifferentColumnsError


class ICAFilter():
    def __init__(self, n_components : int = None):
        '''
            Setting the algorithm
        :param n_components: integer, by default = None
            Number of components to keep
        '''
        self.n_components = n_components

    def filter(self, dataframe : pd.DataFrame, target : str):
        '''
            Creating filter to new data and separating superimposed signals
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param target: string
             The column name of the value that we have to predict
        '''
        # Checking for missing values.
        if dataframe.isna().values.any():
            raise MissingDataError(
                "The passed data frame contains missing values."
            )

        # if a column has string values raise an error
        for col in dataframe:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                raise NonNumericDataError('The given DataFrame contains non-numeric values !')

        # if the given target doesn't match any column in the DataFrame raise an error
        if target not in dataframe:
            raise NoSuchColumnError(
                "The passed target column doesn't exist."
            )

        # Splitting the dataframe.
        self.dataframe = dataframe.copy()
        self.target = target
        self.X_columns = [col for col in self.dataframe.columns if col != self.target]
        self.X = self.dataframe[self.X_columns].values
        self.y = self.dataframe[target].values

        # Creating the filter.
        self.ica = FastICA(n_components = self.n_components)
        self.ica.fit(self.X)

        # Creating new data based on the filter.
        X_ica = self.ica.transform(self.X)
        X_new = self.ica.inverse_transform(X_ica)
        X_new = pd.DataFrame(X_new, columns=self.X_columns)

        # Create and return new Dataframe.
        X_new[target] = self.y
        return X_new

    def apply(self, dataframe : pd.DataFrame):
        '''
            Separating superimposed signals
            based on an already existed filter
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param target: string
             The column name of the value that we have to predict
        '''
        # Checking for missing values.
        if dataframe.isna().values.any():
            raise MissingDataError(
                "The passed data frame contains missing values."
            )

        # if a column has string values raise an error
        for col in dataframe:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                raise NonNumericDataError('The given DataFrame contains non-numeric values !')

        # if the given target doesn't match any column in the DataFrame raise an error
        if self.target not in dataframe:
            raise NoSuchColumnError(
                "The passed target column doesn't exist."
            )

        # Splitting the data.
        self.data_frame = dataframe.copy()

        # Checking if the names of the columns are different for the both dataframes.
        if list(self.data_frame.columns) != list(self.dataframe.columns):
            raise DifferentColumnsError('The passed dataframe has different columns from the one passed to filter function.')
        X_columns = [col for col in self.data_frame.columns if col != self.target]
        X = self.data_frame[X_columns].values
        y = self.data_frame[self.target].values

        # Applying filter to the new dataframe.
        X_ica = self.ica.transform(X)
        X_new = self.ica.inverse_transform(X_ica)
        X_new = pd.DataFrame(X_new, columns=self.X_columns)

        # Create and return new Dataframe.
        X_new[self.target] = y
        return X_new
