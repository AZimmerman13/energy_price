import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import importlib
# matplotlib.use("Agg")
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SKPipe
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.model_selection import GridSearchCV
from src.pipeline import Pipeline
from src.helpers import plot_corr_matrix, scree_plot, plot_num_estimators_mse, gridsearch, pdplots, compare_default_models, pca_with_scree, feat_imp_plots, plot_oob_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __name__ == '__main__':
    df = Pipeline('s3://ajzcap3/spain_data.csv')
    us = Pipeline('data/us_data.csv')

    # make spanish data look more like us data
    drop_cols = ['Valencia_wind_speed', 'Madrid_temp', 'Madrid_wind_speed',
       'Seville_temp', 'Bilbao_temp', 'Bilbao_wind_speed', ' Barcelona_temp', 'generation fossil oil']

    for i in drop_cols:
        df.df.drop(i, inplace=True, axis=1)

    # Combine hydro
    df.df['conventional hydro'] = df.df['generation hydro run-of-river and poundage'] + df.df['generation hydro water reservoir']
    df.df.drop(['generation hydro run-of-river and poundage','generation hydro water reservoir'], inplace=True, axis=1)

    # combine coal
    df.df['coal'] = df.df['generation fossil hard coal'] + df.df['generation fossil brown coal/lignite']
    df.df.drop(['generation fossil hard coal','generation fossil brown coal/lignite'], inplace=True, axis=1)






    df.getXy('price actual')
    df.create_holdout()

    X_train = df.X_train
    y_train = df.y_train
    X_test = df.X_test
    y_test = df.y_test
    X_holdout = df.X_holdout
    y_holdout = df.y_holdout

    #Best Model
    '''
    rf = RandomForestRegressor(max_depth=None, max_features='auto', n_estimators=100, oob_score=True, n_jobs=-1, verbose=True, ccp_alpha=0.0)
    rf.fit(X_train, y_train)
    print("Train R2: ", rf.score(X_train, y_train))
    print("Test R2: ", rf.score(X_test, y_test))


    et = ExtraTreesRegressor(verbose=True, n_jobs=-1)
    et.fit(X_train, y_train)
    print("Train R2: ", rf.score(X_train, y_train))
    print("Test R2: ", rf.score(X_test, y_test))
    '''


    # GHG analysis
    # with no extra conversion, results will be in kg CO2e

    ghg_cols = ['generation biomass', 'generation fossil gas', 'generation nuclear', 'generation solar', 'generation wind onshore', 'conventional hydro', 'coal']

    ipcc_data = [230, 490, 12, 48, 11.5, 24, 820]
    ipcc_cols = ['biomass', 'nat_gas', "nuclear", 'solar_PV_util', "wind", 'hydro', 'coal']
    ipcc = pd.Series(ipcc_data, index=ghg_cols, name=0)

    ghg = df.df[ghg_cols]
    ghg['kg_CO2e'] = ghg.dot(ipcc)
    # df.df['emission'] = ghg.dot(ipcc)
    ghg['price_dollars'] = df.df['price actual'] * 1.12

    fig, ax = plt.subplots()
    ax.scatter(ghg.price_dollars, ghg.kg_CO2e, s=0.2)
    ax.set_title('GHG Emissions as a Function of Energy Price')
    ax.set_xlabel("Price ($)")
    ax.set_yscale('linear')
    ax.set_ylabel("GHG emissions (kg CO2e)")
    plt.savefig('images/ghg_over_price.png')

    # feat_imp_plots(df, rf)
    fig, ax = plt.subplots()
    ax.hist(df.df['total load actual'], bins=60)
    ax.set_title("Distribution of Hourly Electricity Demand")
    ax.set_xlabel('Demand (MW)')
    ax.set_ylabel('Count')
    plt.savefig('images/load_eda.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(df.df['coal'], bins=60)
    ax.set_title("Distribution of Hourly Coal Generation")
    ax.set_xlabel('Generation (MWh)')
    ax.set_ylabel('Count')
    plt.savefig('images/coal_eda.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(df.df['price actual'], bins=60)
    ax.set_title("Distribution of Hourly Electricity Price")
    ax.set_xlabel('Price (EUR/MWh)')
    ax.set_ylabel('Count')
    plt.savefig('images/price_eda.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(df.df['generation wind onshore'], bins=60)
    ax.set_title("Distribution of Hourly Wind Generation")
    ax.set_xlabel('Generation (MWh)')
    ax.set_ylabel('Count')
    plt.savefig('images/wind_eda.png')
    plt.close()

    






   