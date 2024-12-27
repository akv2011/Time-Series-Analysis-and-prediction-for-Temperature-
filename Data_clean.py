import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import plotly.express as px




df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/dataset/New_data/train_city_0.csv')
print(df.dtypes)


df['SO2_concentration'] = pd.to_numeric(df['SO2_concentration'], errors='coerce')
df['O3_concentration'] = pd.to_numeric(df['O3_concentration'], errors='coerce')
df['CO_concentration'] = pd.to_numeric(df['CO_concentration'], errors='coerce')


df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%y %H:%M', errors='coerce')



print(df.dtypes)


print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().any())
print ("\nUnique values :  \n",df.nunique())
     



df.isnull().sum()

df.query('Datetime != Datetime ')



missing_dates_df = df[df['Datetime'].isna()]


before_dates = []
after_dates = []


for idx in missing_dates_df.index:
    
    if idx > 0:
        before_date = df.loc[idx - 1, 'Datetime']
    else:
        before_date = None

    
    if idx < len(df) - 1:
        after_date = df.loc[idx + 1, 'Datetime']
    else:
        after_date = None

    
    before_dates.append(before_date)
    after_dates.append(after_date)


missing_dates_df = missing_dates_df.copy()
missing_dates_df['Before_Datetime'] = before_dates
missing_dates_df['After_Datetime'] = after_dates

print(missing_dates_df[['ID', 'Datetime', 'Before_Datetime', 'After_Datetime']])



time_increment = pd.Timedelta(hours=1)


for i in range(1, len(df)):
    if pd.isna(df.loc[i, 'Datetime']):
        df.loc[i, 'Datetime'] = df.loc[i - 1, 'Datetime'] + time_increment

print(df)




duplicate_mask = df.duplicated('Datetime', keep=False)
duplicates = df[duplicate_mask]

print("Duplicate Timestamps Found:")
print(df[duplicate_mask][['ID', 'Datetime', 'Particulate_matter']].sort_values('Datetime'))


fig = px.line(df, x='Datetime', y='Temperature', title='Temperature_matter with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show(renderer="browser") 



fig = px.line(df, x='Datetime', y='CO_concentration', title='CO_concentration with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show(renderer="browser") 





fig = px.line(df, x='Datetime', y='O3_concentration', title='O3_concentration with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show(renderer="browser") 


fig = px.line(df, x='Datetime', y='Anonymous_X1', title='Anonymous_X1 with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show(renderer="browser") 



df = df.set_index('Datetime')
df.loc['2017-01-01 18:00:00':'2018-02-04 18:00:00']



df_na = df.copy()
df_na=df_na.dropna()     

pd.plotting.autocorrelation_plot(df_na['2017':'2020']['Temperature'])
pd.plotting.autocorrelation_plot(df_na['2017':'2020']['Temperature'].resample("1ME").mean())


pd.plotting.lag_plot(df['Temperature'],lag=1)
pd.plotting.lag_plot(df['Temperature'],lag=3)
         
pd.plotting.lag_plot(df['Temperature'],lag=24) 
pd.plotting.lag_plot(df['Temperature'],lag=8640)



pd.plotting.autocorrelation_plot(df_na['2017':'2020']['Anonymous_X1'])
pd.plotting.autocorrelation_plot(df_na['2017':'2020']['Anonymous_X1'].resample("1ME").mean())

pd.plotting.lag_plot(df['Anonymous_X1'], lag=1)
pd.plotting.lag_plot(df['Anonymous_X1'], lag=3)
pd.plotting.lag_plot(df['Anonymous_X1'], lag=24)
pd.plotting.lag_plot(df['Anonymous_X1'], lag=8640)


pd.plotting.autocorrelation_plot(df_na['2017':'2020']['Particulate_matter'])
pd.plotting.autocorrelation_plot(df_na['2017':'2020']['Particulate_matter'].resample("1ME").mean())

pd.plotting.lag_plot(df['Particulate_matter'], lag=1)
pd.plotting.lag_plot(df['Particulate_matter'], lag=3)
pd.plotting.lag_plot(df['Particulate_matter'], lag=24)
pd.plotting.lag_plot(df['Particulate_matter'], lag=8640)


pd.plotting.autocorrelation_plot(df_na['2017':'2020']['CO_concentration'])
pd.plotting.autocorrelation_plot(df_na['2017':'2020']['CO_concentration'].resample("1ME").mean())

pd.plotting.lag_plot(df['SO2_concentration'], lag=1)
pd.plotting.lag_plot(df['SO2_concentration'], lag=3)
pd.plotting.lag_plot(df['SO2_concentration'], lag=24)
pd.plotting.lag_plot(df['SO2_concentration'], lag=8640)


pd.plotting.autocorrelation_plot(df_na['2017':'2020']['O3_concentration'])
pd.plotting.autocorrelation_plot(df_na['2017':'2020']['O3_concentration'].resample("1ME").mean())

pd.plotting.lag_plot(df['O3_concentration'], lag=1)
pd.plotting.lag_plot(df['O3_concentration'], lag=3)
pd.plotting.lag_plot(df['O3_concentration'], lag=24)
pd.plotting.lag_plot(df['O3_concentration'], lag=8640)








df.columns =df.columns.str.strip()


df['CO_concentration'] = df['CO_concentration'].fillna(method='ffill')
df['Anonymous_X1'] = df['Anonymous_X1'].fillna(method='ffill')


df['Particulate_matter'] = df['Particulate_matter'].interpolate(method='linear')
df['O3_concentration'] = df['O3_concentration'].interpolate(method='linear')


print(df.isnull().sum())

df.info()



fig = px.line(df, x='Datetime', y='Particulate_matter', title='Particulate_matter with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show(renderer="browser") 



fig = px.line(df, x='Datetime', y='CO_concentration', title='CO_concentration with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show(renderer="browser") 




fig = px.line(df, x='Datetime', y='O3_concentration', title='O3_concentration with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show(renderer="browser") 


fig = px.line(df, x='Datetime', y='Anonymous_X1', title='Anonymous_X1 with Slider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show(renderer="browser") 



df = df.set_index('Datetime')
df.loc['2017-01-01 18:00:00':'2018-02-04 18:00:00']




# For Anonymous_X1
pd.plotting.autocorrelation_plot(df['2017':'2020']['Anonymous_X1'])
pd.plotting.autocorrelation_plot(df['2017':'2020']['Anonymous_X1'].resample("1ME").mean())

pd.plotting.lag_plot(df['Anonymous_X1'], lag=1)
pd.plotting.lag_plot(df['Anonymous_X1'], lag=3)
pd.plotting.lag_plot(df['Anonymous_X1'], lag=24)
pd.plotting.lag_plot(df['Anonymous_X1'], lag=8640)

# For Particulate_matter
pd.plotting.autocorrelation_plot(df['2017':'2020']['Particulate_matter'])
pd.plotting.autocorrelation_plot(df['2017':'2020']['Particulate_matter'].resample("1ME").mean())

pd.plotting.lag_plot(df['Particulate_matter'], lag=1)
pd.plotting.lag_plot(df['Particulate_matter'], lag=3)
pd.plotting.lag_plot(df['Particulate_matter'], lag=24)
pd.plotting.lag_plot(df['Particulate_matter'], lag=8640)

# For SO2_concentration
pd.plotting.autocorrelation_plot(df['2017':'2020']['SO2_concentration'])
pd.plotting.autocorrelation_plot(df['2017':'2020']['SO2_concentration'].resample("1ME").mean())

pd.plotting.lag_plot(df['SO2_concentration'], lag=1)
pd.plotting.lag_plot(df['SO2_concentration'], lag=3)
pd.plotting.lag_plot(df['SO2_concentration'], lag=24)
pd.plotting.lag_plot(df['SO2_concentration'], lag=8640)

# For O3_concentration
pd.plotting.autocorrelation_plot(df['2017':'2020']['O3_concentration'])
pd.plotting.autocorrelation_plot(df['2017':'2020']['O3_concentration'].resample("1ME").mean())

pd.plotting.lag_plot(df['O3_concentration'], lag=1)
pd.plotting.lag_plot(df['O3_concentration'], lag=3)
pd.plotting.lag_plot(df['O3_concentration'], lag=24)
pd.plotting.lag_plot(df['O3_concentration'], lag=8640)


print(df.isnull().sum())



###########################################################################


import pandas as pd
import numpy as np
import os




df = pd.read_csv('/home/systemx86/Desktop/Hack/Spy_Zen_aws/Hack_earth/dataset/New_data/Modi_dataset/train_modi_c0.csv')
print(df.dtypes)

df = pd.DataFrame(df)
print(df.describe())

a1=df.describe()




lower_percentile = 0.01
upper_percentile = 0.99


numeric_cols = df.select_dtypes(include=[float, int])


lower_bounds = numeric_cols.quantile(lower_percentile)
upper_bounds = numeric_cols.quantile(upper_percentile)


outliers = pd.DataFrame()
for col in numeric_cols.columns:
    lower_bound = lower_bounds[col]
    upper_bound = upper_bounds[col]
    outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)


outlier_rows = df[outliers.any(axis=1)]
outlier_rows
