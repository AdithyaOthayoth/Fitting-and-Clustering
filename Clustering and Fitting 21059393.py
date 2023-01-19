# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 19:50:34 2023

@author: Adithya O
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import errors as err
import csv
#reading the data set 
def ReadandReturnData(filename):
    
    Data = pd.read_csv(filename,skiprows=4)
    df_Data = pd.DataFrame(Data) 
    df1 = df_Data.loc[df_Data['Country Code'].isin(['GBR'])]
    cols= ['Indicator Name','Indicator Code','Country Code']
    df_dropped = df1.drop(cols, axis=1)
    df1 = df_dropped.reset_index(drop=True).fillna(0.0)
    df1=df1.iloc[0:, 0:56]
    df2 = df1.set_index('Country Name').transpose()
    df2['Years'] = df2.index
    temp_cols = df2.columns.tolist()
    new_cols = temp_cols[-1:] + temp_cols[:-1]
    df2 = df2[new_cols]
    df2 = df2.reset_index(drop = True)
    df2 = df2.rename_axis(None, axis = 1)
    df2=df2.rename(columns={'United Kingdom': 'Electric power consumption (kWh per capita)'})
    df2.plot("Years", "Electric power consumption (kWh per capita)")
    plt.show()
    return df2  
def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f

#reading data set name
df2=ReadandReturnData('Dataset.csv')

print(type(df2["Years"].iloc[1]))
df2["Years"] = pd.to_numeric(df2["Years"])
print(type(df2["Years"].iloc[1]))
param, covar = opt.curve_fit(exponential, df2["Years"], df2["Electric power consumption (kWh per capita)"],
p0=(73233967692.102798, 0.03))

df2["fit"] = exponential(df2["Years"], *param)
df2.plot("Years", ["Electric power consumption (kWh per capita)", "fit"])
plt.show()

# year = np.arange(1960, 2014)
# sigma = np.sqrt(np.diag(covar))
# forecast = exponential(year, *param)
# low, up = err.err_ranges(year, exponential, param, sigma)

# plt.figure()
# plt.plot(df2["Years"], df2["United States"], label="United States")
# plt.plot(year, forecast, label="forecast")
# plt.fill_between(year, low, up, color="yellow", alpha=0.7)
# plt.xlabel("Years")
# plt.ylabel("United States")
# plt.legend()
# plt.show()









