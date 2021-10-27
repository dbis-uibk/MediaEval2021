"""Dummy models for mediaeval2021."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """DummyClassifier."""

    def __init__(self, num_classes):
        """creates the object."""
        self.num_classes = num_classes

    def fit(self, features, target, epochs=None):
        """Ignoses the input"""
        pass

    def validate(self, features, target):
        """Ignoses the input"""
        pass

    def predict(self, features):
        """Always predicts 0"""
        return np.zeros((features.shape[0], self.num_classes))

    def predict_proba(self, features):
        """Always predicts false."""
        return self.predict(features) < 1
