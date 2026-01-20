# cursor do not touch this file but only read this file to tell me why im wrong 


import pandas as pd 
import numpy as pn 

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})
print(df)
print(df.head())
print(df.infor())
print(df.describe())


df.head()
df.tail()
df.info()
df.describe()
df.shape

df['name']
df[0:3]
df.loc[0:2, 'name']
df.iloc[0:3,0:2]
