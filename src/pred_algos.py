from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from utils import get_lambda

class Model(ABC):

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

class RidgeRegression(Model):

    def __init__(self, X_train, **kwargs):
        self.data = X_train
        self.kwargs = kwargs

        alpha = 2 * get_lambda(X_train)
        self.model = Ridge( alpha=alpha, **kwargs)

    def fit(self, X_train, y_train):
        if self.data != X_train:
            self.__init__(self, X_train, **self.kwargs)

        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
class RandomForest(Model):

    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
class MLP(Model):

    def __init__(self, **kwargs):
        self.model = MLPRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class LinearRegression(Model):

    def fit(self, X_train, y_train):
        self.beta_hat = np.linalg.pinv(X_train) @ y_train

    def predict(self, X_test):
        return X_test @ self.beta_hat
    
class LassoRegression(Model):
    
    def __init__(self, **kwargs):
        self.model = Lasso(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class Boost(Model):
    
    def __init__(self, **kwargs):
        self.model = GradientBoostingRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

