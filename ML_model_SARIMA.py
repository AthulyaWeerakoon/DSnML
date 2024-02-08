from typing import Optional

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas import DataFrame

outlet_data = pd.read_csv('train.csv')


def pad_month(date: str):
    mdy = date.split('/')
    for i in range(0, 1):
        if len(mdy[i]) == 1:
            mdy[i] = '0' + mdy[i]
    return mdy[0] + '/' + mdy[1] + '/' + mdy[2]


def rainfall_to_double(rainfall: str):
    return float(rainfall[0: -2])


def extract_outlet_code(outlet: str):
    return int(outlet.split('_')[-1])


def prepare_data(dataset: Optional[DataFrame]):
    dataset['week_start_date'] = dataset['week_start_date'].map(pad_month)
    dataset['week_start_date'] = pd.to_datetime(dataset['week_start_date'], format="%m/%d/%Y")
    dataset['outlet_code'] = dataset['outlet_code'].map(extract_outlet_code)
    dataset = dataset.sort_values(by=['week_start_date', 'outlet_code'], ascending=[True, True])
    print(dataset)
    dataset['expected_rainfall'] = dataset['expected_rainfall'].map(rainfall_to_double)
    dataset = dataset.loc[:, ['expected_rainfall', 'sales_quantity']]
    return dataset


# preparing data
prepared_data = prepare_data(outlet_data)
print(prepared_data)
prepared_data_np = prepared_data.to_numpy()

# restructuring only sales data
sales_data = prepared_data_np[:, 1]
print(sales_data)
sales_data = sales_data.reshape(27, 4200)
sales_data = np.transpose(sales_data)
print(sales_data, sales_data.shape)

# restructuring only rainfall data
rainfall_data = prepared_data_np[:, 0]
print(rainfall_data)
rainfall_data = rainfall_data.reshape(27, 4200)
rainfall_data = np.transpose(rainfall_data)
print(rainfall_data, rainfall_data.shape)

# rebuilding dataframe
sales_df = pd.DataFrame(sales_data, columns=[f'Week_{i+1}' for i in range(27)])
exog_df = pd.DataFrame(rainfall_data, columns=[f'Rainfall_{i+1}' for i in range(27)])

# recreating time index
sales_df['Date'] = pd.date_range(start='2023-01-02', periods=len(sales_df), freq='W')
sales_df.set_index('Date', inplace=True)

exog_df['Date'] = pd.date_range(start='2023-01-02', periods=len(exog_df), freq='W')
exog_df.set_index('Date', inplace=True)

# SARIMAX model with an exogenous variable
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 54)

# preparing exogenous factors
test_data = pd.read_csv('test.csv')
prepared_test_data = prepare_data(test_data).to_numpy()
prepared_exogenous = prepared_test_data[:, 0]
prepared_exogenous = prepared_exogenous.reshape(1, 4200)


model = sm.tsa.SARIMAX(sales_df.mean(axis=1), order=order, seasonal_order=seasonal_order)
results = model.fit(disp=True, method='powell')

# forecast
forecast_steps = 1  # number of steps to forecast
forecast = results.get_forecast(steps=forecast_steps, exog=prepared_exogenous)

# confidence intervals for the forecast
conf_int = forecast.conf_int()

# plot the forecast
ax = sales_df.mean(axis=1).plot(title='Mean Sales Across Outlets with Exogenous Variable')
forecast.predicted_mean.plot(ax=ax, label='Forecast', color='red')
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.2)
plt.legend()
plt.show()


