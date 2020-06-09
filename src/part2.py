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
       'Seville_temp', 'Bilbao_temp', 'Bilbao_wind_speed', ' Barcelona_temp']

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

    df.standardize(X_train)
    X_std = df.X_std

    #Best Model
    random_forest = False
    extra_trees = False
    do_pca = False
    
    if random_forest:
        rf = RandomForestRegressor(max_depth=None, max_features='auto', n_estimators=100, oob_score=True, n_jobs=-1, verbose=True, ccp_alpha=0.0)
        rf.fit(X_train, y_train)
        print("Train R2: ", rf.score(X_train, y_train))
        print("Test R2: ", rf.score(X_test, y_test))
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))}")

    if extra_trees:
        et = ExtraTreesRegressor(verbose=True, n_jobs=-1)
        et.fit(X_train, y_train)
        print("Train R2: ", rf.score(X_train, y_train))
        print("Test R2: ", rf.score(X_test, y_test))
    
    if do_pca:
        pca_with_scree(df)

    

    # GHG analysis
    # with no extra conversion, results will be in kg CO2e

    ghg_cols = ['generation biomass', 'generation fossil gas', 'generation nuclear', 'generation solar', 'generation wind onshore', 'conventional hydro', 'coal']

    ipcc_data = [820, 490, 230, 48, 24, 12, 11.5]
    ipcc_cols = ['Coal', 'Natural Gas', 'Biomass', 'Solar', 'Hydro', "Nuclear", "Wind"]
    ipcc = pd.Series(ipcc_data, index=ghg_cols, name=0)
    graph_ipcc = False
    if graph_ipcc:
        fig, ax = plt.subplots()
        ax.bar(ipcc_cols, ipcc_data)
        ax.set_title("IPCC Global Warming Potential", fontsize=20)
        ax.set_xlabel("Source", fontsize=14)
        ax.set_ylabel("grams CO2e / kWh", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('images/ipcc_bar.png')
        plt.close()



    ghg = df.df[ghg_cols]
    ghg['kg_CO2e'] = ghg.dot(ipcc)
    # df.df['emission'] = ghg.dot(ipcc)
    ghg['price_dollars'] = df.df['price actual'] * 1.12

    plot_ghg_over_price = True
    if plot_ghg_over_price:
        fig, ax = plt.subplots()
        ax.scatter(ghg.price_dollars, ghg.kg_CO2e, s=0.2)
        ax.set_title('GHG Emissions as a Function of Energy Price', fontsize=14)
        ax.set_xlabel("Price ($)", fontsize=13)
        ax.set_yscale('linear')
        ax.set_ylabel("GHG emissions (kg CO2e)", fontsize=13)
        plt.savefig('images/ghg_over_price.png')
        plt.close()

    # feat_imp_plots(df, rf)
    plot_eda=False
    if plot_eda:
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

    # Make hourly EDA plot
    df.df['hour'] = df.df.index.hour
    hourly = df.df.groupby('hour').mean()
    misc = hourly['generation biomass'] + hourly['generation fossil oil'] + hourly['generation waste']
    nuclear = hourly['generation nuclear']
    ng = hourly['generation fossil gas'] + hourly['generation nuclear']
    coal = ng + hourly['coal']
    hydro = coal + hourly['conventional hydro']
    solar = hydro + hourly['generation solar']
    wind = solar + hourly['generation wind onshore']
    imports = hourly['total load actual']

    stacked = [nuclear, ng, coal, hydro, solar, wind, imports]
    labels = ["Nuclear", "Natural Gas", "Coal", 'Conventional Hydro', 'Solar', 'Wind', 'Imports']
    colors = ['palegreen', 'red', 'black', 'darkgreen', 'orange', 'purple', 'brown']
    make_hourly=True
    if make_hourly:
        fig, ax = plt.subplots()
        ax.set_title("Hourly Electricity Demand", fontsize=20)
        ax.set_ylabel('Average Demand (MW)', fontsize=14)
        ax.set_xlabel('Hour of Day', fontsize=14)
        # ax.plot(hourly.index, hourly['total load actual'], label='Total Demand', color='brown')
        
        

        for (source, label, color) in zip(stacked, labels, colors):
            ax.plot(hourly.index, source, color=color, label=label)

        ax.fill_between(hourly.index, nuclear, 4000, color='palegreen', alpha=0.6)
        ax.fill_between(hourly.index, ng, nuclear, color='red', alpha=0.2)
        ax.fill_between(hourly.index, coal, ng, color='black', alpha=0.2)
        ax.fill_between(hourly.index, hydro, coal, color='darkgreen', alpha=0.2)
        ax.fill_between(hourly.index, solar, hydro, color='orange', alpha=0.2)
        ax.fill_between(hourly.index, wind, solar, color='purple', alpha=0.2)
        ax.fill_between(hourly.index, hourly['total load actual'], wind, color='brown', alpha=0.2)
        ax.plot(hourly.index, hourly['total load actual'], label='Total Demand')
        

        # plt.legend()
        handles,labels = ax.get_legend_handles_labels()
        # plt.tight_layout()
        plt.savefig('images/hourly_demand_and_sources.png')
        plt.close()

        need_legend = False
        if need_legend:
            fig, axe = plt.subplots()
            axe.legend(handles, labels, loc='center')
            axe.xaxis.set_visible(False)
            axe.yaxis.set_visible(False)
            plt.savefig('images/legend.png')
            plt.close()

    # Make bar plot from IPCC table
    spain_pcts = [40, 20, 19, 14, 5, 3]
    spain_labels = ['Coal & NG', 'Nuclear', 'Wind', 'Hydro', 'Solar', 'Biomass']
    us_pcts = [63, 19, 6, 7, 2, 2]
    us_labels = ['Coal & NG', 'Nuclear', 'Wind', 'Hydro', 'Solar', 'Biomass']
   
    mix=True
    if mix:
        fig, ax = plt.subplots()
        ax.bar(spain_labels, spain_pcts)
        plt.xticks(rotation=45)
        ax.set_title('Spain Electricity Generation Mixture', fontsize=20)
        ax.set_ylabel('Percent of Total Generation', fontsize=14)
        ax.set_xlabel('Generation Source', fontsize=15)
        plt.tight_layout()
        plt.savefig('images/spain_mixture.png')
        plt.close()

        fig, ax = plt.subplots()
        ax.bar(us_labels, us_pcts)
        plt.xticks(rotation=45)
        ax.set_title('U.S. Electricity Generation Mixture', fontsize=20)
        ax.set_ylabel('Percent of Total Generation', fontsize=14)
        ax.set_xlabel('Generation Source', fontsize=15)
        plt.tight_layout()
        plt.savefig('images/us_mixture.png')
        plt.close()
    
        fig, ax = plt.subplots()
        width = 0.35
        x = np.arange(len(us_labels))
        rects1 = ax.bar(x - width/2, spain_pcts, width, label='Spain')
        rects2 = ax.bar(x + width/2, us_pcts, width, label='U.S.')

        ax.set_ylabel('Percent of Total Generation', fontsize=14)
        ax.set_xlabel('Source', fontsize=15)
        ax.set_title('U.S. vs. Spain Electricity Mixture', fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(us_labels)
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/combo_mix.png')
        plt.close()






    






   