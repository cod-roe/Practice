#%%
import pandas as pd
import numpy as np
# %%
s = [50,60,70,80,90]
sr = pd.Series(s)
sr
# %%
type(sr)
# %%
sr.dtype
# %%
sr.index
# %%
sr.index.to_list()
# %%
sr.index.values
# %%
sr.index = ['a','b','c','d','e']
sr
# %%
sr.index
# %%
d = {'a':50,'b':60,'c':70,'d':80,'e':90}
sr = pd.Series(d)
sr
# %%
print(sr.values)
# %%
sr['a']
# %%
sr[0]
# %%
sr[1:4]
# %%
sr[::-1]
# %%
sr[[1,3]]
# %%
sr[['a','c']]
# %%
sr.tolist()
# %%
list(sr)
# %%
tuple(sr)
# %%
sr2 = pd.Series([10])
sr2
# %%
sr= pd.concat([sr,sr2])
sr
# %%
sr[sr < 50]
# %%
sr[sr == 50]

# %%
sr_str = sr.astype(str)
sr_str
# %%
sr_str.str[0:1]
# %%
sr_str.str.contains('7')
# %%
sr_str[sr_str.str.contains('7')]
# %%
sr_str = sr_str.reset_index(drop=True)
sr_str
# %%
sr_str[sr_str.str.contains('\d')]
# %%
print(sr_str[sr_str.str.contains('\d',regex=False)])

# %%
test = pd.Series([50,60,'山本',80,90,'田中'])
test
# %%
test.str.contains('田中',regex=False)
# %%
test.str.contains('田中',na=False,regex=False)
# %%
test[test.str.contains('田中',na=False,regex=False)]
# %%
df = pd.DataFrame([[1,2,3],[1,2,3]])
df
# %%
df.columns = ['A','B','C']
df
# %%
df.index = ['1行','2行']
df
# %%
df.T
# %%
df['A']
# %%
df.A
# %%
df[['A','C']]
# %%
df.columns
# %%
new_df = pd.DataFrame([[4,5,6]],index=['3行'],columns=df.columns)
new_df
# %%
_df = pd.concat([df,new_df],axis=0)
_df
# %%
_df.reset_index(drop=True,inplace=True)
# %%
_df
# %%
_df['D'] = _df['B'] + _df['C']
_df
# %%
_df.describe()
# %%
_df.describe().round(2)
# %%
_df.mean()
# %%
_df.max()
# %%
_df.count()
# %%
_df.info()
# %%
_df.isna().sum()
# %%
_df.isna()
# %%
file_path = '/tmp/work/src/input/other/htemp.csv'
df_htemp = pd.read_csv(file_path,encoding='cp932')
df_htemp.head()
# %%
df_htemp.tail()
# %%
df_htemp.columns
# %%
df_htemp2 = df_htemp[['都道府県','地点', '現在時刻(年)', '現在時刻(月)', '現在時刻(日)', '3日の最高気温(℃)']]
df_htemp2
# %%
df_htemp3 = df_htemp[-5:]
df_htemp3
# %%
df_htemp3.dtypes
# %%
df_htemp.shape
# %%
df_htemp.index
# %%
df_htemp.iloc[5:9,2:10]
# %%
df_htemp['地点'] =='旭川（アサヒカワ）'
# %%
df_htemp['地点'].str.contains('旭川（アサヒカワ）')
# %%
df_htemp[df_htemp['地点'] =='旭川（アサヒカワ）']
# %%
(df_htemp['地点'] =='旭川（アサヒカワ）').mean()
# %%
df_htemp.query('地点 == "旭川（アサヒカワ）"')
# %%
df_htemp[df_htemp['地点'].isin(['旭川（アサヒカワ）'])]
# %%
df_htemp[(df_htemp['地点'] =='旭川（アサヒカワ）') | (df_htemp['地点'] =='札幌（サッポロ）')]



# %%
df_htemp.query('地点 == "旭川（アサヒカワ）"|地点 == "札幌（サッポロ）"')
# %%
df_htemp['地点'].unique()

# %%
city_df = df_htemp[
  (df_htemp['地点'] =='札幌（サッポロ）') |
  (df_htemp['地点'] =='東京（トウキョウ）') |
  (df_htemp['地点'] =='横浜（ヨコハマ）') |
  (df_htemp['地点'] =='名古屋（ナゴヤ）') |
  (df_htemp['地点'] =='大阪（オオサカ）') |
  (df_htemp['地点'] =='福岡（フクオカ）') |
  (df_htemp['地点'] =='那覇（ナハ）') ]
city_df
#%%
city_df = city_df[city_df.columns[[1,2,4,5,6,9]]]

# %%
city_df
# %%
city_df.to_csv('export.csv',index=False)
# %%
city_df.to_csv('export.csv',index=False,encoding='shift_jis')
# %%
df = pd.read_csv('export.csv',encoding='shift_jis')
# %%
df
# %%
df.columns = ['都道府県','地点','年','月','日','最高気温']
# %%
df
# %%
df = df.rename(columns={'地点':'都市'},)
df
# %%
df.sort_values('最高気温',ascending=False)
# %%
