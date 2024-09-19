#%%
print('Hello World')
# %%
import pandas as pd
import numpy as np
# %%
# データ作成
s = pd.Series([1,2,3,4],index={'a','b','c','d'})
s
# %%
data = {
  'Name':['Alice','Bob','Charlie','David','Eva'],
  'Age':[25,30,35,40,45],
  'Product':['Apple','Banana','Cherry','Peach','Grape']
}
df = pd.DataFrame(data)
df
# %% 演習
data2 = {
  '名前':['佐藤','鈴木','高橋'],
  '身長':[175.0,161.8,165.4]
}
df2 = pd.DataFrame(data2)
df2
# %% データ読み込み
data_path = '/tmp/work/src/input/titanic/'
df3 = pd.read_csv(data_path + 'train.csv')
df3

# %%
df3.shape
# %%
df3.dtypes
# %%
df3.info()
# %%
df3.head()
# %%
df3.tail()
# %%
df3.sample(5)
# %%
df3.head()
# %%
df3[['PassengerId','Name','Age']].head()
# %%
df3.sample(frac=0.01)
# %%
df3.at[1, 'Name']
# %%
df3.loc[3]
# %%
df3.loc[10:20]
# %%
df3.loc[10:20,['Name','Age']]

# %%
df3.iloc[10:21,[3,5]]
# %%
df3.loc[31:44,['PassengerId','Pclass','Ticket']]
# %% データの絞り込み
df3[df3['Age'] >=30]

# %%
df3.query('Age >=30')

# %%
df = pd.read_csv(data_path + 'train.csv')


# %%
df[(df['Age'] >=30 & (df['Pclass'] == 1))]
# %%
df[(df['Age'] >=30) & (df['Pclass'] == 1)]


# %%
df[(df['Age'] >=30) | (df['Pclass'] == 1)]

# %%
df[(df['Embarked'] == 'S') & (df['Fare'] >= 50)]
# %%
df.dtypes
# %%
df['PassengerId_str'] = df['PassengerId'].astype(str)
# %%
df.head()
# %%
df.dtypes
# %%
df['PassengerId_int16'] =  df['PassengerId'].astype(np.int16)
# %%
df.dtypes
# %%
df['Pclass'].value_counts()
# %%
df['Pclass_int8']= df['Pclass'].astype(np.int8)
# %%
df.dtypes
# %%
df['Pclass_int8'].value_counts()
# %%
df['Pclass'].nunique()
# %%
df['Age'].isna().sum()
# %%
df['Age'].isna().mean()

# %%
df2 = df.dropna(subset='Age')
df2.isna().sum()
# %%
df['Age'].fillna(40)
# %%
df['Age']
# %%
df['Embarked'].isna().sum()
# %%
df['Embarked'].notna().sum()

# %%
df2['Embarked'] = df['Embarked'].fillna('S')
df2['Embarked'].unique()

# %%
df2['Embarked'].isna().sum()
# %%
print(df['Fare'].sum())
print(df['Fare'].mean())
print(df['Fare'].max())
print(df['Fare'].min())
# %%
df['Fare'].describe()
# %%
df.describe()
# %%
df['Fare'].describe(percentiles=[0.25,0.75,0.9])

# %%
df['SibSp_Parch'] = df['SibSp'] + df['Parch']
# %%
df.head(3)
# %%
df.drop(columns=['SibSp_Parch'])
# %%
df.head()
# %%
df2 = df.drop(columns=['SibSp_Parch'])
# %%
df2.head()
# %%
df.drop(0)
# %%
df2 = df2.rename(columns={'PassengerId_str':'PassengerId_string'})
# %%
df2.head(3)
# %%
df2 = df2.drop(columns=['Pclass_int8'])
df2['Fare_twice']  = df2['Fare'] * 2
df2.head(3)
# %%
df.drop_duplicates(subset=['Survived','Pclass'])
# %%
df.drop_duplicates()

# %%
df.sort_values('Fare')
# %%
df.sort_values('Fare',ascending=False)

# %%
df2 = df2.drop([0,2])
df2.head()
# %%
df2.reset_index(drop=True)
# %%
q8 = df.drop_duplicates(subset=['Embarked'])
q8.head()
# %%
q8.reset_index(drop=True)
# %%
q8.shape
# %%
def is_over_thirty(row):
  if row >= 30:
    return 1
  else:
    return 0
  

# %%
df['Age'].map(is_over_thirty)
# %%
df['is_over30'] = df['Age'].map(is_over_thirty)

# %%
df[['Age','is_over30']].head()
# %%
lambda row: 1 if row >= 30 else 0
# %%
df['is_over30_lambda'] = df['Age'].map(lambda row: 1 if row >= 30 else 0
)

# %%
df[['Age','is_over30','is_over30_lambda']].head()

# %%
df['is_over30'].replace({1:100,0:-100})
# %%
df['Name'].str.upper()
# %%
df['Name'].str.lower()
# %%
pd.get_dummies(df['Embarked'],dtype=np.int8,prefix='Embarked')
# %%
pd.get_dummies(df['Sex'],dtype=np.int8,drop_first=True).rename(columns={'male':'is_male'})
# %%
df.columns
# %%
df['Ticket'].value_counts()
# %%
df['Ticket'].nunique()
# %%
df['Pclass'].unique()
# %%
df['Embarked'].unique()
# %%
df['Embarked'].isna().sum()
# %%
df['Pclass'].value_counts()

# %%
df['Embarked'].isin(['A'])
# %%
df['Embarked'].isin(['A']).sum()

# %%
df['Embarked'].isin(['S']).sum()

# %%
df['Embarked'].value_counts()

# %%
print(df['Survived'].unique())
df['Survived'].value_counts().sort_values(ascending=False)

# %%
df1 = pd.DataFrame({
  'employee':[1,2,3,4],
  'name':['Alice','Bob','Charlie','David'],
  'department':['HR','Engineering','Marketing','Sales']
})
df1
# %%
df2 = pd.DataFrame({
  'employee':[2,3,4,5],
  'salary':[60000,80000,70000,90000],
  'manager_id':[1,2,1,4],
})
df2
# %%
mergerd_df = pd.merge(df1,df2,on='employee',how='inner')
mergerd_df
# %%
mergerd_df2 = pd.merge(df1,df2,on='employee',how='outer')
mergerd_df2
# %%
df1 = pd.DataFrame({
  'A':['A0','A1','A2'],
  'B':['B0','B1','B2'],
  'C':['C0','C1','C2'],
  'D':['D0','D1','D2'],
})

df2 = pd.DataFrame({
  'A':['A3','A4','A5'],
  'B':['B3','B4','B5'],
  'C':['C3','C4','C5'],
  'D':['D3','D4','D5'],
})

# %%
df1
# %%
df2
# %%
concat_df = pd.concat([df1,df2],axis=0)
concat_df
# %%
concat_df.reset_index(drop=True,inplace=True)
# %%
concat_df
# %%
products_df = pd.DataFrame({
  'product_id':[1,2,3,4,5],
  'product_name':['スマートフォン','ノートパソコン','タブレット','イヤホン','スピーカー'],
  'price':[50000,100000,30000,10000,20000],
})

stock_df = pd.DataFrame({
  'product_id':[1,2,4,5,7],
  'quantity':[100,50,30,200,80],
})
# %%
concat_df2 = pd.merge(products_df,stock_df,on='product_id',how='left')
concat_df2
# %%
df.groupby('Pclass')['Age'].agg('mean')
# %%
df.groupby('Pclass')['Age'].mean()

# %%
df.groupby('Pclass').agg({'Age':['mean','median'],'Fare':['mean']})
# %%
df.columns

# %%
df.groupby(['Pclass','Embarked'])['Age'].mean()

# %%
df.crosstab(['Pclass','Embarked'])
# %%
pd.crosstab(df['Pclass'],df['Embarked'])
# %%

# %%
df.groupby(['Pclass','Embarked'])['Age'].mean().unstack()

# %%
df.pivot_table(index='Pclass',columns='Embarked',values='Age',aggfunc='mean')
# %%
df.columns
# %%
df.head(1)
# %%
df.pivot_table(index='Sex',columns='Pclass',values='Fare',aggfunc='mean')
# %%
df['Pclass'].plot(kind='bar')
# %%
df['Pclass'].value_counts().sort_index().plot(kind='bar')
# %%
df['Age'].plot(kind='hist')
# %%
df['Age'].plot(kind='box')
# %%
df['Age'].median()
# %%
df['Fare'].plot(kind='hist')
# %%
df['Embarked'].value_counts()
# %%
df['Embarked'].value_counts().plot(kind='bar')

# %%
df['Fare'].plot(kind='box')

# %%
df[df['Fare'] >= 100][['Fare']] = 100
# %%
df['Fare_hensin'] = df['Fare'].map(lambda x:x if x < 100 else 100)
df['Fare_hensin'].plot(kind='box')
# %%
df['Fare_hensin'].plot(kind='hist')

# %%
output_dir = '/tmp/work/src/input/'
df.to_csv(output_dir + 'titanic_data.csv',index=False)
# %%
import numpy as np

# %%
np.random.seed(0)

date_range = pd.date_range(start='2024-01-01',end='2024-3-31',freq='D')
price = np.random.uniform(130,150,len(date_range))

df_time = pd.DataFrame({
  'Date':date_range,
  'Price':price
})

df_time.head()
# %%
df_time2 = df_time.set_index('Date')
df_time2
# %%
df_time2['2024-01-01':'2024-01-31']
# %%
df_time2['2024-01-01':'2024-01-31']['Price'].mean()

# %%
df_time2['2024-01-01':'2024-01-31'].mean()

# %%
df_time2.resample('M').mean()
# %%
df_time2.resample('M').max()

# %%
df_time2.shift(1)
# %%
df_time2['前日比'] =  df_time2 / df_time2.shift(1)
# %%
df_time2
# %%
df_time2['前日値'] =  df_time2['Price'].shift(1)

# df_time2['前日差'] =  df_time2['prce'] - df_time2['前日値']

df_time2.head()
# %%
df_time2['前日差'] =  df_time2['Price'] - df_time2['前日値']
df_time2.head()

# %%
df_time2.plot(kind='line')
# %%
df_time2['3SMA'] =  df_time2['Price'].rolling(window=3).mean()
df_time2
# %%
df_time2.plot(kind='line')

# %%
df_time2['7SMA'] = df_time2['Price'].rolling(window=7).mean()
df_time2
# %%
df_time2.plot(kind='line')

# %%
