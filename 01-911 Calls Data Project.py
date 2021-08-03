
# For this project, I will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
#
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)

# Essentially, everything was done on Jupyter. However, for saving purpose, I decided to download my code into a Python file.

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('911.csv')  # reading our 911 record data.


# Some basic data analysis
df.info()  # Taking a peek at what is inside the data

df.head()

df['title'].nunique()  # Looking at how many cases are in the dataset.


# Looking at top 5 unique zipcodes for 911 calls
df['zip'].value_counts().head(5)


# Looking at top 5 unique townships for 911 calls.
df['twp'].value_counts().head(5)


# Creating new features

# In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.
# For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

# Fiding most common Reason for a 911 call based off of this new column above.

df['title'].apply(lambda title: title.split()[0]).value_counts()


# Now use seaborn to create a countplot of 911 calls by Reason.

sns.set_style('whitegrid')  # clear the plot first
sns.countplot(x=df['Reason'], palette='viridis')


# analysis on time information

type(df['timeStamp'][0])  # Checking out the current type of timestamp

# Changing timestamps to datetime objects.
converted = pd.to_datetime(df['timeStamp'])

# grabbing specific attributes from the converted datetime objects above.
df['Hour'] = converted.apply(lambda x: x.hour)
df['Month'] = converted.apply(lambda x: x.month)
df['Day'] = converted.apply(lambda x: x.dayofweek)


# Day is in the form of integer (0-6).
# By using map() with a dictionary, convert the integer form into the actual string names to the day of the week

dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['Day'] = df['Day'].map(dmap)


# Applying seaborn to create a countplot of the Day column with hue being the Reason column.

sns.countplot(x=df['Day'], hue=df['Reason'], palette='viridis')


# Same Procedure for Month
sns.countplot(x=df['Month'], hue=df['Reason'], palette='viridis')


# The current data is missing some months in the month column.
# We can fill in this information by plotting the information in Pandas. (simple line plot that fills in the missing info)
# First, create a gropuby object by the month column.
# Then, use the count() method for aggregation.
df.groupby(df['Month']).count()
byMonth = df.groupby(df['Month']).count()
byMonth['twp'].plot()  # See if this fills in the missing info. Works fine!

# ** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[228]:


sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())


# ######

# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. **

# In[229]:


df['Date'] = converted.apply(lambda x: x.date())


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[230]:


byDate = df.groupby(by='Date').count()
byDate['lat'].plot()
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[260]:


byTraffic = df[df['Reason'] == 'Traffic']
byTraffic_and_Date = byTraffic.groupby(by='Date').count()
byTraffic_and_Date['lat'].plot()
plt.tight_layout()
plt.title('Traffic')


# In[199]:


# In[259]:


byFire = df[df['Reason'] == 'Fire']
byFire_and_date = byFire.groupby(by='Date').count()
byFire_and_date['twp'].plot()
plt.tight_layout()
plt.title('Fire')


# In[201]:


# In[252]:


byEMS = df[df['Reason'] == 'EMS']
byEMS_and_Date = byEMS.groupby(by='Date').count()['lat'].plot()
plt.title('EMS')
plt.tight_layout()


# In[202]:


# In[261]:


df


# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[290]:


matrix1 = df.groupby(by=['Day', 'Hour']).count()['Reason'].unstack(level=-1)
matrix1.head()


# In[297]:


sns.heatmap(matrix1, cmap='viridis')


# ** Now create a HeatMap using this new DataFrame. **

# In[303]:


fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(matrix1, cmap='viridis')


# In[204]:


# ** Now create a clustermap using this DataFrame. **

# In[314]:


sns.clustermap(matrix1, cmap='viridis', figsize=(8.5, 8.5))


# In[205]:


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[319]:


matrix2 = df.groupby(by=['Day', 'Month']).count()['Reason'].unstack(level=-1)
matrix2.head()


# In[207]:


# In[325]:


fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(data=matrix2, cmap='viridis')


# In[208]:


# In[328]:


sns.clustermap(data=matrix2, figsize=(8, 7), cmap='viridis')


# In[209]:


# **Continue exploring the Data however you see fit!**
# # Great Job!
