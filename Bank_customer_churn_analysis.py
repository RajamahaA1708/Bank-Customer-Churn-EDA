import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dataset load
df=pd.read_csv("C:\\Users\\PoovethaN\\Downloads\\archive (10)\\Churn_Modelling.csv")

#first five Row
df.head()
print(df.head())

#last five rows
df.tail()
print(df.tail())

#column names, datatypes,
df.info()
print(df.info())

#statistical summary(mean,median)
df.describe()
print(df.describe())

#missing values count
print(df.isnull().sum())

#Dataset shape & info
print('Rows and Columns:', df.shape)
print('\nMissing Values:\n', df.isnull().sum())

#remove unneccessary column making prediction
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

#verify the cleaned dataset
df.head()
# 1.how many members churn with this(Exited: 0 vs 1)
plt.figure(figsize=(5, 4))
sns.countplot(x='Exited', data=df)
plt.title('Customer Churn Count')
plt.show()

# 2. Age vs Churn 
plt.figure(figsize=(5, 4))
sns.boxplot(x='Exited', y='Age', data=df)
plt.title('Age vs Churn')
plt.show()

# 3. Country-wise Churn (France, Spain, Germany)
plt.figure(figsize=(5, 4))
sns.countplot(x='Geography', hue='Exited', data=df)
plt.title('Churn by Geography')
plt.show()

