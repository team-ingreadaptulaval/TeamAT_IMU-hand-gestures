from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


class DimensionalityReducer( object ):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, x, y):
        raise NotImplementedError( "Should have implemented this" )

    def transform(self, x):
        raise NotImplementedError( "Should have implemented this" )


class PCAReducer( DimensionalityReducer ):

    def __init__(self, n_components):
        super().__init__(n_components)
        self.reducer = PCA(n_components=n_components)

    def fit(self, x, y, n_componnents=None):
        if n_componnents is None:
            self.reducer.fit(x, y)
        else:
            self.reducer = PCA(n_components=n_componnents)
            self.reducer.fit(x, y)

    def transform(self, x):
        return self.reducer.transform(x)


class RandomForestReducer( DimensionalityReducer ):

    def __init__(self, n_components):
        super().__init__(n_components)
        self.reducer = RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1)
        self.bests = np.array([])


    def fit(self, x, y):
        self.reducer.fit(x, y)
        importances = self.reducer.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.bests = indices[0:self.n_components]

    def transform(self, x):
        return x[:, self.bests]

class SelectorReducer( DimensionalityReducer ):

    def __init__(self, components):
        super().__init__(len(components))
        self.components = np.array(components)

    def fit(self, x, y):
        pass

    def transform(self, x):
        return x[:, self.components]

class NoReductor(DimensionalityReducer):

    def __init__(self):
        super().__init__(None)

    def fit(self, x, y):
        pass

    def transform(self, x):
        return x
