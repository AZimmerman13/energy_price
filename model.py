import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SKPipe
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.model_selection import GridSearchCV
from src.pipeline import Pipeline
from src.helpers import plot_corr_matrix, scree_plot, plot_num_estimators_mse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __name__ == '__main__':
    full_df = Pipeline('s3://ajzcap3/spain_data.csv')

    X_train = full_df.X_train
    y_train = full_df.y_train
    X_test = full_df.X_test
    y_test = full_df.y_test
    X_holdout = full_df.X_holdout
    y_holdout = full_df.y_holdout

    #Best Model
    rf = RandomForestRegressor(max_depth=None, max_features='auto', n_estimators=30, oob_score=True, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("Train R2: ", rf.score(X_train, y_train))
    print("Test R2: ", rf.score(X_test, y_test))