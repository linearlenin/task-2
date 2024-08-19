#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Unemployment in India.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[7]:


df = df.dropna()
df


# In[8]:


df.isnull().sum()


# In[9]:


df.head()


# In[10]:


print(df.columns)


# In[11]:


df.columns = df.columns.str.strip()


# In[12]:


df.head()


# In[16]:


df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')


# In[17]:


# Define the period for COVID-19
start_date = '2019-12-01'
end_date = '2021-12-31'

# Filter the DataFrame
df_covid = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plot
plt.figure(figsize=(12, 6))

# Plot unemployment rate over time
sns.lineplot(data=df_covid, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.title('Unemployment Rate Over Time During COVID-19')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()


# In[19]:


# Define pre and post-pandemic periods
pre_pandemic = df[(df['Date'] < start_date)]
post_pandemic = df[(df['Date'] > end_date)]

# Calculate average unemployment rates
pre_avg_unemployment = pre_pandemic['Estimated Unemployment Rate (%)'].mean()
post_avg_unemployment = post_pandemic['Estimated Unemployment Rate (%)'].mean()

print(f'Average Unemployment Rate Before Pandemic: {pre_avg_unemployment:.2f}%')
print(f'Average Unemployment Rate After Pandemic: {post_avg_unemployment:.2f}%')


# In[21]:


print(region_avg.head())


# In[22]:


# Group by region and calculate average unemployment rate
region_avg = df_covid.groupby('Region')['Estimated Unemployment Rate (%)'].mean().reset_index()

print(region_avg.head())


# In[23]:


# Check for missing values
print(region_avg.isna().sum())

# Remove rows with missing values if any
region_avg = region_avg.dropna()


# In[24]:


print(region_avg.columns)


# In[26]:


sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=region_avg)
plt.xlabel('Region')
plt.ylabel('Average Unemployment Rate (%)')
plt.title('Average Unemployment Rate by Region During COVID-19')
plt.xticks(rotation=90)
plt.show()


# In[ ]:




