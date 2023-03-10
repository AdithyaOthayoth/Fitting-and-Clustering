# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 19:50:34 2023

@author: Adithya O
"""
# Importing the required libraries.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize 

#reading the data set and returns the dataframe
def ReadandReturnData(filename, indicator):
    """
    The ReadandReturnData function is used to read the filename,
    return a transposed dataframe df2, with Years as column1 and Electric
    power consumption (kWh per capita) as column2 
    and plot the relationship between them.
    """
    Data = pd.read_csv(filename, skiprows=4)
    df_Data = pd.DataFrame(Data) 
    df1 = df_Data.loc[df_Data['Country Code'].isin(['GBR'])]
    cols = ['Indicator Name', 'Indicator Code', 'Country Code']
    df_dropped = df1.drop(cols, axis = 1)
    df1 = df_dropped.reset_index(drop = True).fillna(0.0)
    df1 = df1.iloc[0:, 0:56]
    df2 = df1.set_index('Country Name').transpose()
    df2['Years'] = df2.index
    temp_cols = df2.columns.tolist()
    new_cols = temp_cols[-1:] + temp_cols[:-1]
    df2 = df2[new_cols]
    df2 = df2.reset_index(drop = True)
    df2 = df2.rename_axis(None, axis = 1)
    df2 = df2.rename(columns={'United Kingdom': indicator})
    
    #plotting the data
    df2.plot("Years", "Electric power consumption (kWh per capita)")
    plt.xlabel("Years")
    plt.ylabel("Electric power consumption")
    plt.title("Electric power consumption in United Kingdom")
    plt.show()
    
    #returns data frame
    return df2  

#A function to find the expontial function
def exponential(t, n0, g):
    """
    Calculates exponential function with scale factor n0 and growth rate g.
    """
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper  

#Passing the dataset name to the funtion ReadandReturnData 
#Storing the dataframe in the variable 
df_clusters = ReadandReturnData('Population Growth.csv',
                                "Electric power consumption (kWh per capita)")
df2 = ReadandReturnData('Dataset.csv', "Electric power consumption (kWh per capita)")


#Code for fitting starts here

df2["Years"] = pd.to_numeric(df2["Years"])
#passing parameters to curve_fit and finding the param and covar
param, covar = opt.curve_fit(exponential, df2["Years"],
                             df2["Electric power consumption (kWh per capita)"],
p0=(73233967692.102798, 0.03))

#adding column fit in df2 using the exponential funtion
df2["Fit"] = exponential(df2["Years"], *param)

#plotting the graph after fitting
df2.plot("Years", ["Electric power consumption (kWh per capita)", "Fit"])
plt.xlabel("Years")
plt.ylabel("Electric power consumption")
plt.title("Electric power consumption in United Kingdom")
plt.show()

#passing parameters to err_ranges() function to find the lower and upper range
year = np.arange(1960, 2015)
sigma = np.sqrt(np.diag(covar))
forecast = exponential(year, *param)
low, up = err_ranges(year, exponential, param, sigma)

#plotting the confidence range after finding the lower and upper range from err_ranges().
plt.figure()
plt.plot(df2["Years"], df2["Electric power consumption (kWh per capita)"],
         label="Electric power consumption (kWh per capita)")
plt.plot(year, forecast, label="Fit")
plt.fill_between(year, low, up, color="yellow", alpha=0.7,label='Confidence range')
plt.xlabel("Years")
plt.ylabel("Electric power consumption")
plt.title("Electric power consumption in United Kingdom")
plt.legend()
plt.show()

#Code for fitting ends here


#Code for Clustering starts here
df_clusters = pd.DataFrame(df_clusters)
#Scale data using StandarScaler()
scaler = StandardScaler() 
df_clusters = scaler.fit_transform(df_clusters)

#Normalize the data using normalize()
df_clusters = normalize(df_clusters)   
df_clusters = pd.DataFrame(df_clusters)
x = df_clusters[0]
y = df_clusters[1]

# setting up the clusterer. Used AffinityPropagation for clustering
ap = AffinityPropagation(max_iter=2000)

#runnig it
#fitting data
ap.fit(df_clusters)
labels = ap.labels_
cen = ap.cluster_centers_

# print the number of centres formed
print("\n number of cluster centres", len(cen))

# plot using the labels to select colour
plt.figure(figsize = (5.0, 5.0))
ap_y_pred = ap.predict(df_clusters)
col = ["blue", "red", "green"]
for l in range(0, len(cen)): # loop over the different labels
    plt.scatter(x, y,c=ap_y_pred, cmap='coolwarm')

# show cluster centres
for ic in range(len(cen)):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, marker='*', color='k', markersize=15)
plt.xlabel("Years")
plt.ylabel("Population Growth")
plt.title("Population Growth in United Kingdom")
plt.show()

#Code for clustering ends here




