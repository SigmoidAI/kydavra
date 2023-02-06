'''
Created with love by Sigmoid.
@Author - Spînu Lavinia - laviniaspinu01@gmail.com
'''
# Import all needed libraries.
import pandas as pd
from sklearn.decomposition import FastICA
from .errors import MissingDataError, NonNumericDataError, NoSuchColumnError, NotBetweenZeroAndOneError, NoSuchMethodError, DifferentColumnsError


class ICAReducer():

    def __init__(self, min_corr : float = 0.5, max_corr : float = 0.8, correlation_type : str = "pearson"):
        '''
            Setting up the algorithm
        :param min_corr: float, between 0 and 1, default = 0.5
            The minimal positive correlation value that must the feature have with target
        :param max_corr: float, between 0 and 1, default = 0.8
            The maximal positive correlation value that must the feature have with target
        :param correlation_type: str, default = "pearson"
        '''
        if min_corr < 0 or min_corr > 1:
            raise NotBetweenZeroAndOneError("min_corr isn't between 0 and 1")
        else:
            self.min_corr = min_corr
        if max_corr < 0 or max_corr > 1:
            raise NotBetweenZeroAndOneError("max_corr isn't between 0 and 1")
        else:
            self.max_corr = max_corr
        if correlation_type in ['pearson', 'kendall', 'spearman']:
            self.correlation_type = correlation_type
        else:
            raise NoSuchMethodError(
                f"kydavra doesn't sustain such method as {correlation_type}/nTry 'pearson', 'kendall' or 'spearman'")

    def index_to_cols(self, index_list):
        '''
            Converting the list of indexes in a list of features that will be picked by the model
        :param index_list: list
            A list of Indexes that should be converted into a list of feature names
        :return: list
            A list with feature names
        '''
        return [self.X_columns[i] for i in index_list]


    def reduce(self,dataframe : pd.DataFrame, target : str):
        '''
            Reducing the dimensionality of the data
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
        self.y_column = target

        #Setting variables
        self.dataframe = dataframe.copy()
        self.X_columns = [col for col in self.dataframe.columns if col != target]
        correlated_indexes = []

        #Getting index of label column
        target_index = list(self.dataframe.columns).index(target)

        #Creating correlation table
        self.corr_table = self.dataframe.corr()
        corr_matrix = self.corr_table.values

        #Choosing most important columns
        for i in range(len(corr_matrix[target_index])):
            if abs(corr_matrix[target_index][i]) > self.min_corr and abs(corr_matrix[target_index][i]) < self.max_corr:
                correlated_indexes.append(i)
        self.correlated_cols = self.index_to_cols(correlated_indexes)

        #Creating correlation table for selected columns
        correlated_features_df = self.dataframe[self.correlated_cols]
        correlated_features_matrix = correlated_features_df.corr()
        self.correlated_features = []

        #Selecting most important correlations between features and creting lists of correlations
        for i in range(len(correlated_features_matrix)):
            cols = []
            for j in range(len(correlated_features_matrix)):
                # I also select the col on diagonal because it is also a part of correlation and should be in my array
                if abs(correlated_features_matrix[self.correlated_cols[i]][self.correlated_cols[j]]) > self.min_corr and abs(correlated_features_matrix[self.correlated_cols[i]][self.correlated_cols[j]]) < self.max_corr :
                    cols.append(self.correlated_cols[j])
            if len(cols) != 0 :
                cols.append(self.correlated_cols[i])
                cols.sort()
                self.correlated_features.append(cols)

        #Cleaning selected columns and ordering them by length
        self.correlated_features = [list(item) for item in set(tuple(row) for row in self.correlated_features)]
        self.correlated_features = sorted(self.correlated_features, key=len, reverse=True)
        self.dictionary = {}
        self.correlated_columns = []

        #Creating main dictionary with ICA filters for different correlations
        for i in range(len(self.correlated_features)):
            flag = False
            key = tuple(self.correlated_features[i])
            for j in range(len(self.correlated_features[i])):
                if self.correlated_features[i][j] not in self.dataframe.columns:
                    flag = True
            if flag:
                continue

            #Creating algorithm for every correlation
            self.correlated_columns.append(self.correlated_features[i]) # addding really changed columns
            correlated_features_ICA_df = self.dataframe[self.correlated_features[i]].values
            ica = FastICA(n_components = 1)
            ica.fit(correlated_features_ICA_df)
            X_ica = ica.transform(correlated_features_ICA_df)

            #Creating new column and adding it to dataframe
            column_name = "_".join(key)
            self.dataframe[column_name] = X_ica
            self.dictionary[column_name] = ica

            #Deleting correlated columns
            for j in range(len(self.correlated_features[i])):
                self.dataframe = self.dataframe.drop(self.correlated_features[i][j],1)
        return self.dataframe

    def apply(self, dataframe : pd.DataFrame):
        '''
            Reducing the dimensionality of the data
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

        # Splitting the data.
        self.dataframe = dataframe.copy()

        # Checking if the names of the columns are different for the both dataframes.
        if list(self.data_frame.columns) != list(self.dataframe.columns):
            raise DifferentColumnsError('The passed dataframe has different columns from the one passed to filter function.')
        for i in range(len(self.correlated_columns)):
            correlated_features_ICA_df = self.dataframe[self.correlated_columns[i]].values
            column_name = "_".join(self.correlated_columns[i])
            ica = self.dictionary[column_name]
            X_ica = ica.transform(correlated_features_ICA_df)
            self.dataframe[column_name] = X_ica
            for j in range(len(self.correlated_features[i])):
                self.dataframe = self.dataframe.drop(self.correlated_columns[i][j],1)
        return self.dataframe
