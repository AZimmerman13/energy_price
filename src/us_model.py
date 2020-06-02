import pandas as pd
import numpy as np




if __name__ == '__main__':
    # units = median grams CO2eq/kWh
    ipcc_data = np.array([820, 490, 230, 48, 41, 38, 27, 24, 12, 12, 11]).reshape(1,-1)
    ipcc_cols = ['Coal', 'Natural_gas', 'Biomass', 'Solar_PV_util', 'Solar_PV_res', 'Geothermal', 'CSP', 'Hydro', "Wind_offshore", "Nuclear", "Wind_onshore"]
    IPCC_df = pd.DataFrame(data=ipcc_data, columns=ipcc_cols)
    # https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
    # data is usually presented for 3 or 4 months prior
    price_df = pd.DataFrame()
    price_19 = []
    price_18 = []
    price_17 = []
    price_16 = []
    price_15 = []
    price_14 = []
    price_13 = []
    price_12 = []
    price_11 = [9.62, 9.70]
    price_10 = [9.34, 9.52]

    # test_df = pd.read_csv('data/jan.csv')

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    years = [12, 13, 14, 15, 16, 17, 18, 19]


    sheets = []
    for y in reversed(years):
        for m in reversed(months):
            sheets.append(f"{m}{y}")


    gen_cols = ['coal', 'petrol_liquid', 'petrol_coke', 'nat_gas', 'other_gas', 'Nuclear', 'std_hydro', 'renewable_minus_hydro', 'wind', 'solar', 'wood', 'biomass', 'geothermal', 'hydro_pumped_storage', 'price', 'demand', 'idx']

    # price units = cents / kWh
    # Generation units = thousand MWh
    # Demand = million kWh
    generation = pd.DataFrame(columns=gen_cols)
    errors = []
    for i, sheet in enumerate(sheets):
        if i < 53:
            try:
                df = pd.read_csv(f'data/{sheet}.csv')
                vals =  df.iloc[6:20,2].values
                val = pd.Series(vals, index=gen_cols[:-3])
                generation.loc[i] = val

                generation.price[i] = df.iloc[55, 7]
                generation.demand[i] = df.iloc[55,1]
                generation.idx[i] = sheet
            except:
                errors.append(sheet)
                continue
        elif 53 < i < 71:
            try:
                df = pd.read_csv(f'data/{sheet}.csv')
                vals =  df.iloc[6:20,2].values
                val = pd.Series(vals, index=gen_cols[:-3])
                generation.loc[i] = val

                generation.price[i] = df.iloc[51, 7]
                generation.demand[i] = df.iloc[51,1]
                generation.idx[i] = sheet
            except:
                errors.append(sheet)
                continue
        elif 71 < i < 75:
            try:
                df = pd.read_csv(f'data/{sheet}.csv')
                vals =  df.iloc[7:21,2].values
                val = pd.Series(vals, index=gen_cols[:-3])
                generation.loc[i] = val

                generation.price[i] = df.iloc[56, 7]
                generation.demand[i] = df.iloc[56,1]
                generation.idx[i] = sheet
            except:
                errors.append(sheet)
                continue
        else:
            gen_idx = [7, 8, 9, 10, 11, 12, 13, 14, 19, 18, 15, 16, 17, 20]
            try:
                df = pd.read_csv(f'data/{sheet}.csv')
                vals =  df.iloc[gen_idx,2].values
                val = pd.Series(vals, index=gen_cols[:-3])
                generation.loc[i] = val

                generation.price[i] = df.iloc[56, 7]
                generation.demand[i] = df.iloc[56,1]
                generation.idx[i] = sheet
            except:
                errors.append(sheet)
                continue

    nump = generation.to_numpy()
    # us_data = pd.DataFrame(nump, index=sheets, columns = gen_cols)
    us_data = generation.set_index('idx')
    us_data.dropna(how='any', inplace=True)

    for col in us_data.columns:
        us_data[col] = [float(i.replace(',','')) for i in us_data[col]]

    us_data.to_csv('data/us_data.csv')

    # months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    # years = [2018, 2019]


    # sheets = []
    # for y in reversed(years):
    #     for m in reversed(months):
    #         sheets.append(f"{m}{y}")

    # for i in sheets:
    #     pd.read_csv(f'data/test/{i}.csv')
    #     pd.to_csv(f'data/test/{i}test.csv')