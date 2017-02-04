
# coding: utf-8

# In[79]:

import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#http://data.un.org/Explorer.aspx
# Create a DataFrame using pandas read_csv()
df = pd.read_csv('C:\\Users\\ryan\Desktop\\Thinkful DS Sample Data - Prep Course\\Sample Data\\UNdata_Export_20170201_062423700.csv')


# In[140]:

# Take a quick look at the data structure
df.head()


# In[155]:

# Isolate USA and plot CO2 emissions over time (1990-2013)
df2 = df.ix[lambda df2: df['Country or Area'] == 'United States of America', :]
plt.plot(df2['Year'], df2['Value'], color = 'red')
plt.xlabel('Year')
plt.ylabel('Kilotonnes CO2 Emitted')
plt.title('CO2 Emission Trend in the United States of America 1990-2013')
plt.show()


# In[99]:

# Find minimum and maximum years of data by country
print(df.groupby('Country or Area').min())
print(df.groupby('Country or Area').max())


# In[156]:

# Some countries only have data through 2012. Hence I will exclude 2013 for consistency.
df3 = df.ix[lambda df3: df['Year'] <= 2012, :]
# Aggregate all countries included in the dataset and plot.
df4 = df3.groupby('Year').sum()
df4['Value']
plt.plot(df4['Value'])
plt.xlabel('Year')
plt.ylabel('Kilotonnes CO2 Emitted')
plt.title('Global CO2 Emission Trend (Where UN Data Available) 1990-2012')
plt.show()


# In[137]:

# Compare aggregate CO2 distributions of 1990 with 2012.
df5 = df.ix[lambda df: df['Year'] == 1990, :]
df6 = df.ix[lambda df: df['Year'] == 2012, :]
plt.hist(df5['Value'], color = 'red', alpha = 0.5)
plt.hist(df6['Value'], color = 'blue', alpha = 0.5)


# In[152]:

# Show a box-plot of the CO2 emissions for 2012
x = df6['Value']
# Convert column to row array for use in boxplot() function
x_transpose = []
for val in x:
    x_transpose.append(val)
plt.boxplot(x_transpose)


# In[ ]:



