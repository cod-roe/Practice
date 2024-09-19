#%%
import pandas as pd
import numpy as np
# %%
input_dir = '/tmp/work/src/input/titanic/'
df =  pd.read_csv(input_dir + 'train.csv')
# %%
df.head(10)
# %%
df.shape
# %%
df.columns
# %%
df['Name']
# %%
df[['Name','Sex']]
# %%
df[df['Sex'] == 'male'].head(10)
# %%
df.loc[df['Sex'] == 'male'].head()
# %%
df.query('Sex == "male"').head()
# %%
df.sort_values('Age',ascending=False).head(5)
# %%
df.groupby('Pclass')['Fare'].agg('mean')
# %%
def is_20(row):
  if row < 20:
    return '20歳未満'
  elif row >= 20:
    return '20歳以上'
  
df['AgeGroup'] = df['Age'].map(is_20)
df[['Age','AgeGroup']].head(10)
# %%
df['AgeGroup2'] = df['Age'].map(lambda x:'20歳以上' if x >= 20 else '20歳未満' )
df[['Age','AgeGroup','AgeGroup2']].head(10)


# %%
df.describe()
# %%
df[['Age','Fare']].corr()
# %%
df['Age'].plot(kind='hist')
# %%
df.plot(kind='scatter',x='Age',y='Fare')
# %%
