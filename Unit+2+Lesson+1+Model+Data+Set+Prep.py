
# coding: utf-8

# In[75]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#
df = pd.read_csv('C:\\Users\\ryan\\Desktop\\Thinkful DS Sample Data - Main Course\\Unit 2\\credit_card_defaults.csv', header = 1)


# In[22]:

df.head()


# In[26]:

#Select a subset of features to analyze

df2 = pd.DataFrame()
df2 = df.loc[:, ['EDUCATION', 'AGE', 'LIMIT_BAL', 'BILL_AMT1', 'PAY_AMT1', 'default payment next month']]


# In[27]:

df2.head()


# In[38]:

df2.EDUCATION.unique()
#Education must be between 1-4 according to data descriptions --> DROP 0,5,6. 4 = 'OTHER', but since it is unknown it will
#obfuscate results. DROP 4 as well

df2 = df2[(df2['EDUCATION'] > 0) & (df2['EDUCATION'] < 4)]


# In[36]:

g = sns.PairGrid(df2.dropna(), diag_sharey=False)
# Scatterplot.
g.map_upper(plt.scatter, alpha=.5)
# Fit line summarizing the linear relationship of the two variables.
g.map_lower(sns.regplot, scatter_kws=dict(alpha=0))
# Give information about the univariate distributions of the variables.
g.map_diag(sns.kdeplot, lw=3)
plt.show()


# In[40]:

# Make the correlation matrix.
corrmat = df2.corr()
print(corrmat)

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn.
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# In[73]:

#Plot categorical w/ continuous using box-plots

get_ipython().magic('matplotlib inline')

# Making a four-panel plot.

sns.boxplot(x = df2['EDUCATION'], y = df2['AGE'])
plt.show()

sns.boxplot(x = df2['EDUCATION'], y = df2['LIMIT_BAL'])
plt.show()

sns.boxplot(x = df2['EDUCATION'], y = df2['PAY_AMT1'])
plt.show()

sns.boxplot(x = df2['EDUCATION'], y = df2['BILL_AMT1'])
plt.show()


# In[74]:

# Descriptive statistics by group.
print(df2.groupby('EDUCATION').describe())


# In[ ]:




# In[84]:

import scipy.stats as stats
df3 = pd.DataFrame()
df3 = df2[['EDUCATION', 'default payment next month']]

# Test whether education differences are signficant in terms of default rate

print(df3.groupby('EDUCATION').mean())
print('Graduate School (1) vs. University (2)')
print(stats.ttest_ind(df3[df3.EDUCATION == 1]['default payment next month'] , 
                      df3[df3.EDUCATION == 2]['default payment next month']))

print('Graduate School (1) vs. High School (3)')
print(stats.ttest_ind(df3[df3.EDUCATION == 1]['default payment next month'] , 
                      df3[df3.EDUCATION == 3]['default payment next month']))

print('University (2) vs. High School (3)')
print(stats.ttest_ind(df3[df3.EDUCATION == 2]['default payment next month'] , 
                      df3[df3.EDUCATION == 3]['default payment next month']))


# In[ ]:

# All 3 levels of schooling have different means, with a p < 0.05, although University vs High School had a p value 0.045

#Two categorical variable visuals (and corresponding chi-square test) not included because age is ordinal, not categorical -- too many categories 


# In[85]:

#Now for feature engineering 

df2.head()


# In[111]:

#FEATURES 1-3

#Feature 1: bill_amt1 / Limit_bal. Useful because we are seeing what % of available credit is being used - credit hungry?
df2['feat1'] = df2['BILL_AMT1'] / df2['LIMIT_BAL']

#Feature 2: pay_amt1 / bill_amt1. Useful because we are seeing what % of the bill the person pays that month
df2['feat2'] = df2['PAY_AMT1'] / df2['BILL_AMT1'] 

#Feature 3: Group ages together, as more datapoints in a group should prevent defaults from skewing rates in small populations

# Set a default value
df2['feat3'] = '0'
# Set Age_Group value for all row indexes which Age is LT 18
df2['feat3'][df2['AGE'] <= 18] = 'LTE 18'
# Same procedure for other age groups
df2['feat3'][(df2['AGE'] > 18) & (df2['AGE'] <= 30)] = '19-30'
df2['feat3'][(df2['AGE'] > 30) & (df2['AGE'] <= 40)] = '31-40'
df2['feat3'][(df2['AGE'] > 40) & (df2['AGE'] <= 50)] = '41-50'
df2['feat3'][(df2['AGE'] > 50) & (df2['AGE'] <= 60)] = '51-60' 
df2['feat3'][(df2['AGE'] > 60) & (df2['AGE'] <= 70)] = '61-70'
df2['feat3'][(df2['AGE'] > 70)] = '70+'
    
df2.head()


# In[129]:

# FEATURES 4-6

fig = plt.figure()

fig.add_subplot(221)
plt.hist(df2['LIMIT_BAL'].dropna())
plt.title('Raw')

fig.add_subplot(222)
plt.hist(np.log(df2['LIMIT_BAL'].dropna()))
plt.title('Log')

fig.add_subplot(223)
plt.hist(np.sqrt(df2['LIMIT_BAL'].dropna()))
plt.title('Square root')

ax3=fig.add_subplot(224)
plt.hist(1/df2['LIMIT_BAL'].dropna())
plt.title('Inverse')
plt.tight_layout()
plt.show()

#Feature 4: Log fcn looks more normal than 'raw' for LIMIT_BAL; will use it as a feature for test purposes
df2['feat4'] = np.log(df2['LIMIT_BAL'].dropna())


#Feature 5: Sqrt fcn looks more normal than 'raw' for LIMIT_BAL; will use it as a feature for test purposes
df2['feat5'] = np.sqrt(df2['LIMIT_BAL'].dropna())


#Feature 6: Available credit minus used credit
df2['feat6'] = df2['LIMIT_BAL'] - df2['BILL_AMT1']


# In[130]:

df2.head(50)


# In[133]:

# FEATURE 7

#Feature 7: Will try the ratio of pay_amt1 / limit_bal to see if high payments relative to credit line affect the outcome
df2['feat7'] = df2['PAY_AMT1'] / df2['LIMIT_BAL']


# In[139]:

#FEATURES 8, 9, and 10 will take the continuous numerical variables for LIMIT_BAL, PAY_AMT1, and BILL_AMT1 and normalize them
from sklearn import preprocessing

# Select only numeric variables to scale.
df_num = df2.loc[:, 'LIMIT_BAL' : 'PAY_AMT1'].select_dtypes(include=[np.number]).dropna()

# Save the column names.
names=df_num.columns

# Scale, then turn the resulting numpy array back into a data frame with the
# correct column names.
df_scaled = pd.DataFrame(preprocessing.scale(df_num), columns=names)

# The new features contain all the information of the old ones, but on a new scale.
#plt.scatter(df_num['LIMIT_BAL'], df_scaled['LIMIT_BAL'])
#plt.show()

# Lookit all those matching means and standard deviations!
#print(df_scaled.describe())


df2['feat8'] = df_scaled['LIMIT_BAL']
df2['feat9'] = df_scaled['BILL_AMT1']
df2['feat10'] = df_scaled['PAY_AMT1']

df2.head()


# In[142]:

df2.corr()

#Top 5 absolute corr values for default indicator are:

#feat4
#feat5
#LIMIT_BAL
#feat6
#feat1



# In[132]:

## Ask about this issue -- what are the error messages telling me?

fig = plt.figure()

fig.add_subplot(221)
plt.hist(df2['PAY_AMT1'].dropna())
plt.title('Raw')

fig.add_subplot(222)
plt.hist(np.log(df2['PAY_AMT1'].dropna()))
plt.title('Log')

fig.add_subplot(223)
plt.hist(np.sqrt(df2['PAY_AMT1'].dropna()))
plt.title('Square root')

ax3=fig.add_subplot(224)
plt.hist(1/df2['PAY_AMT1'].dropna())
plt.title('Inverse')
plt.tight_layout()
plt.show()


# In[ ]:




# In[ ]:



