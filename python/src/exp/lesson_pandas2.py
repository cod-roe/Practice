#%%
import pandas as pd
import numpy as np
from IPython.core.display import display
# %%
df_sample = pd.DataFrame({
  'A':['a','b','c','d','e'],
  'B':['aa','bb','cc','dd','ee'],
  'C':['aaa','bbb','ccc','ddd','eee'],
  'D':['aaaa','bbbb','cccc','dddd','eeee'],
})
# %%
df_sample
# %%
input_dir = '/tmp/work/src/input/titanic/'
df = pd.read_csv(input_dir + 'train.csv')
df.head()
# %%
df.to_csv('pandas_train2.csv',index=False)
# %%
df['Satisfied'] = 0
display(df['Satisfied'].head())
df['Satisfied'].sum()
# %%
df['AgeBin'] = pd.cut(df['Age'],4)
df.head()
# %%
df['AgeBin'].unique()
# %%
df['AgeBin'].nunique()
# %%
df['FareBin'] = pd.qcut(df['Fare'],4)
df['FareBin'].value_counts()
# %%
df['Fare'].describe()

#%%
df['Age'].isna().sum()


# %%
df.fillna({'Age':df['Age'].median()},inplace=True)
df['Age'].isna().sum()

# %%
df['Embarked'].value_counts()

# %%
df['Embarked'].isna().sum()

# %%
df.fillna({'Embarked':df['Embarked'].mode()[0]}, inplace=True)
df['Embarked'].isna().sum()

# %%
df['Embarked'].mode()[0]
# %%
df['Embarked'].value_counts()
# %%
type(df['Embarked'].mode())
# %%
df.drop(['PassengerId','Ticket'],inplace=True,axis=1)
display(df.head())
df.columns
# %%
df['FamilySize'] = df['SibSp'] + df['Parch'] +1
df[['SibSp','Parch','FamilySize']].tail(5)
# %%
df.loc[:,['SibSp','Parch','FamilySize']]
# %%
df['IsAlone'] = 1

df['IsAlone'].loc[df['FamilySize'] > 1] = 0

df.loc[:,['FamilySize','IsAlone']]

# %%
df = pd.get_dummies(df,columns=['Sex','Embarked'],dtype=np.int8)
# %%
pd.get_dummies(df['Sex'],dtype=np.int8)

# %%
df_1 = df.drop(columns=['Satisfied'],axis=1)
df_2 = df[['Satisfied','Name']]
display(df_1.head())
display(df_2.head())

# %%
df_1.shape
# %%
df = pd.read_csv(input_dir + 'train.csv')

# %%
df.shape
# %%
df.columns
# %%
df.info()
# %%
df.isna().mean()
# %%
df.describe()
# %%
df.describe(include='O')
# %%
df.describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9])
# %%
df['Sex'].value_counts()
# %%
display(df['Survived'].value_counts())
display(df['Survived'].value_counts()/len(df['Survived']))

# %%
pd.crosstab(df['Sex'],df['Survived'])
# %%
pd.crosstab(df['Sex'],df['Survived'],normalize='columns')

# %%
df['Survived'] == 0
# %%
df.loc[df['Survived'] == 0]

# %%
df.loc[df['Survived'] == 1,['Age']].dropna(subset='Age')

# %%
df.head(5)
# %%
df.sample(5)
# %%
df['Age']
# %%
df[['Age','Name']]
# %%
df.loc[:,['Age','Name']]
# %%
df.iloc[df.index % 2 == 1]
# %%
df.iloc[lambda x : x.index % 2 == 1]

# %%

