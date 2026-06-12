#!/usr/bin/env python3
"""Module to standardize tabular data"""
from sklearn import preprocessing


def Standardize(X):
    """Standardizes tabular data using Scikit-learn"""
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
