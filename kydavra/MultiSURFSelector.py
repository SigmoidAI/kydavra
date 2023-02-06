'''
Created with love by Sigmoid
@Author - Basoc Nicoleta-Nina - nicoleta.basoc28@gmail.com
'''

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .errors import MissingDataError, NonNumericDataError, NoSuchColumnError


class MultiSURFSelector:

    def __init__(self, n_neighbours: int = 7, n_features: int = 5) -> None:
        """
            Setting up the algorithm
        :param n_neighbors: int, default = 7
            The number of neighbors to consider when assigning feature
            importance scores. If a float number is provided, that percentage of
            training samples is used as the number of neighbors.
            More neighbors results in more accurate scores, but takes longer.
            
        :param n_features: int, default = 5
            The number of top features (according to the relieff score) to
            retain after feature selection is applied.
        """
        
        self.n_neighbours = n_neighbours
        self.n_features = n_features
        
    def define_tresh_multisurf(self, dataframe : pd.DataFrame, target : str) -> float:
        """
            Defining the treshold for the multiSURF algorithm
        :param dataframe: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param target: str
             The column name of the value that we what to predict
        :return: float
            
        """
        
        X=dataframe.drop([target], axis=1).values
        distance_vector=[]
        # Defining the distance array based on which we will calculate the threshold.
        for i in range(len(X)):
            for j in range(len(X)):
                if i!=j:
                    distance_vector.append(np.linalg.norm(X[i]-X[j]))
        
        # Calculating the average distances and their stds and the value of the threshold.
        average_distance=np.array(distance_vector).mean()
        std_vector=np.std(distance_vector)/2
        diference_tresh_std=average_distance - std_vector
        return diference_tresh_std
        
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
            
        # if the given target doesn't match any column in the DataFrame raise an error
        if target not in dataframe:
            raise NoSuchColumnError(f'No such column {target}')
            
       # if a column has string values raise an error
        for col in dataframe:
            if not pd.api.types.is_numeric_dtype(dataframe[col]):
                raise NonNumericDataError('The given DataFrame contains non-numeric values')
            
        # Converting the data frame in numpy arrays.
        X = dataframe.drop([target], axis=1).values
        y = dataframe[target].values
        
        # Obtaining the number of features and samples.
        n_samples, feature_number = X.shape
        
        # Initialising the score of all features as 0.
        score = np.zeros(feature_number)
        
        # Using NearestNeighbors to define the nighbours for our target instance and all others.
        nbrs = NearestNeighbors(n_neighbors=n_samples,algorithm='kd_tree').fit(X)
        
        # Getting the distances of the neighbours and their indices.
        distances, indices = nbrs.kneighbors(X)
        
        # Getting how many classes we have to predict. 
        n_classes = np.unique(y).tolist()
        
        self.treshold=self.define_tresh_multisurf(dataframe, target)
        
        # Identifying hits and misses.
        for i in range(n_samples):
            
            # Initialising the miss and hit counters. 
            miss=dict()
            hit=[]
            
            # Getting all the classes of the target column.
            for j in n_classes:
                miss[j] = []
            
            # Finding misses and hits based on the threshold. 
            for n in range(1, self.n_neighbors):
                if np.linalg.norm(y[i] - y[indices[i][n]])<self.treshold:
                    if y[i] == y[indices[i][n]]:
                        hit.append(indices[i][n])
                    else:
                        miss[y[indices[i][n]]].append(indices[i][n])
            
            # Feature Weight update based on the right misses and hits.
            cont = dict()
            cont_diff_idx = float(len(y[y == y[i]])) / float(n_samples)
            
            # Computing the probabilities of every class.
            for label in n_classes:
                cont_diff_class = float(len(y[y == label])) / float(n_samples)
                cont[label] = cont_diff_class/ (1 - cont_diff_idx)
                
            # For hits.
            hit_term = np.zeros(feature_number)
            for h in hit:
                hit_term = np.array(hit_term) + np.array(abs(X[i, :] - X[h, :]))
                
            # For misses.
            miss_term = dict()
            for (label, miss_list) in miss.items():
                miss_term[label] = np.zeros(feature_number)
                for m in miss_list:
                    miss_term[label] = np.array(abs(X[i, :] - X[m, :])) + np.array(miss_term[label])
                
                # Update score for miss.
                score += miss_term[label] / (self.n_neighbors * n_samples)
                
            # Update score for hit.
            score -= hit_term / self.n_neighbors

        # Sorting the indexes based on the score.    
        result = np.argsort(score, 0)
        index = result[::-1][:self.n_features]
        
        # Return the columns based on their score.
        return list(dataframe.iloc[:, index].columns)                           
