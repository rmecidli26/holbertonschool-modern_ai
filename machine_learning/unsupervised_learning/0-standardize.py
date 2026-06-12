#!/usr/bin/env python3
from sklearn import preprocessing
import numpy as np

def Standardize(X):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
