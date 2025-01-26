# -*- coding: utf-8 -*-
"""
Created on Sun March 26 13:14:34 2024

@author: niloufar
"""
# visualizations.py

import matplotlib.pyplot as plt

def plot_variogram(lags, gamma, model_gamma, title="Variogram"):
    """
    Plot experimental and modeled variograms.

    Parameters:
        lags (np.ndarray): Lag distances.
        gamma (np.ndarray): Experimental semivariance.
        model_gamma (np.ndarray): Modeled semivariance.
        title (str): Plot title.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(lags, gamma, 'o', label="Experimental Variogram")
    plt.plot(lags, model_gamma, '-', label="Fitted Model")
    plt.xlabel("Lag Distance")
    plt.ylabel("Semivariance")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
