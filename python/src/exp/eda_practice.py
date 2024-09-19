#%%
print('test')
# %% ライブラリ読み込み
import numpy as np
import pandas as pd
import os
import pickle
import gc
from IPython.display import display
#分布確認
import ydata_profiling as php

#可視化
import matplotlib.pyplot as plt
import seaborn as sns

#前処理
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

#モデリング
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

import japanize_matplotlib 
#!%matplotlib inline 
sns.set(font='IPAexGothic') 
# %%
file_path = '/tmp/work/input/'

#ファイルの読み込み
df_train = pd.read_csv(file_path + 'train.csv')
df_train.head()



# %% データセット 説明変数と目的変数
x_train, y_train, id_train = df_train[['Pclass','Fare']],\
df_train[['Survived']],\
df_train[['PassengerId']]

print(x_train.shape, y_train.shape, id_train.shape)


#%% モデル学習・評価の関数定義
#モデル学習 クロスバリデーション
params = {
  'boosting_type': 'gbdt',
  'objective': 'binary',
  'metric': 'auc',
  'learning_rate': 0.1,
  'num_leaves': 16,
  'n_estimators': 100000,
  'random_state': 123,
  'importance_type': 'gain'
}

def train_cv(input_x,
             input_y,
             input_id,
             params,
             n_splits=5,
):
  metrics = []
  imp = pd.DataFrame()

  cv =list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(input_x, input_y))


  for nfold in np.arange(n_splits):
    print('-'*20, nfold, '-'*20)
    idx_tr, idx_va = cv[nfold][0], cv[nfold][1] 
    x_tr, y_tr = input_x.loc[idx_tr, :], input_y.loc[idx_tr, :]
    x_va, y_va = input_x.loc[idx_va, :], input_y.loc[idx_va, :]
    print(x_tr.shape, y_tr.shape)
    print(x_va.shape, y_va.shape)
    print(f"y_train:{input_y['Survived'].mean():.3f}, y_tr:{y_tr['Survived'].mean():.3f}, y_va:{y_va['Survived'].mean():.3f}")
    
    model = lgb.LGBMClassifier(**params)
    model.fit(x_tr,
              y_tr,
              eval_set=[(x_tr,y_tr), (x_va,y_va)],
              callbacks=[
              lgb.early_stopping(stopping_rounds=100, verbose=True),
              lgb.log_evaluation(100),
              ]
              )
    
    y_tr_pred = model.predict(x_tr)
    y_va_pred = model.predict(x_va)
    metric_tr = accuracy_score(y_tr, y_tr_pred)
    metric_va = accuracy_score(y_va, y_va_pred)
    print(f'[accuracy] tr:{metric_tr:.2f}, va:{metric_va:.2f} ')
    metrics.append([nfold, metric_tr, metric_va])

    _imp = pd.DataFrame({'col':input_x.columns,
    'imp':model.feature_importances_,
    'nfold':nfold})
    imp = pd.concat([imp, _imp], axis=0, ignore_index=True)


  print('='*20, 'result', '='*20)
  metrics = np.array(metrics)
  print(metrics)


  print(f"[cv] tr:{metrics[:,1].mean():.2f}+-{metrics[:,1].std():.2f}, va: {metrics[:,2].mean():.2f}+-{metrics[:,2].std():.2f}")

  imp = imp.groupby('col')['imp'].agg(['mean','std'])
  imp.columns = ['imp', 'imp_std']
  imp = imp.reset_index(drop=False)

  print('Done.')

  return imp, metrics



# %% 
# 関数を用いてモデル学習
imp, metrics = train_cv(x_train, y_train, id_train, params,n_splits=5)
#%%
x_train, y_train, id_train = df_train[['Pclass','Fare', 'Age']],\
  df_train[['Survived']],\
    df_train[['PassengerId']]
#%%
print(x_train.shape, y_train.shape, id_train.shape)

#%%
imp, metrics = train_cv(x_train, y_train, id_train, params,n_splits=5)


# %%
imp.sort_values('imp', ascending=False, ignore_index=True)


# %% 
# データ前処理
df_train.describe().T
# %%
df_train.describe(include='O').T
# %%
df_train[['Fare']].agg(['mean']).T
# %%
df_train[['Fare']].agg(['mean','std','min','max']).T

# %%
df_train[['Fare']].agg(['dtype','count','nunique','mean','std','min','max']).T

# %%
df_train.agg(['dtype','count','nunique']).T
# %%
df_train['Sex'].value_counts()
# %%
# php.ProfileReport(df_train)
# %%
# profile_report = php.ProfileReport(df_train)

# %%
# profile_report.to_file('report.html')
# %% 欠損値
df_train.isna().sum()

# %% 欠損値0埋め
df_train['Age_fillna_0'] = df_train['Age'].fillna(0)
df_train.loc[df_train['Age'].isna(), ['Age', 'Age_fillna_0']].head()
# %% 平均値埋め
df_train['Age_fillna_mean'] = df_train['Age'].fillna(df_train['Age'].mean())

df_train.loc[df_train['Age'].isna(), ['Age', 'Age_fillna_mean']].head()
# %% 空白埋め
df_train['Cabin_fillna_space'] = df_train['Cabin'].fillna('')
df_train.loc[df_train['Cabin'].isna(), ['Cabin', 'Cabin_fillna_space']].head()
# %%
df_train['Cabin_fillna_mode'] = df_train['Cabin'].fillna(df_train['Cabin'].mode()[0])

df_train.loc[df_train['Cabin'].isna(), ['Cabin', 'Cabin_fillna_mode']]
# %% 
# 外れ値
df_train['Age'].agg(['min', 'max'])
# %%
#!%matplotlib inline 


# %%
sns.histplot(data=df_train['Age'])

# %%
df_train['Age'].hist()
# %%
quartile = df_train['Age'].quantile(q=0.75) - df_train['Age'].quantile(q=0.25)
print('四分位範囲:',quartile )
print('下限値:',df_train['Age'].quantile(q=0.25) - quartile*1.5)
print('上限値:',df_train['Age'].quantile(q=0.75) - quartile*1.5)
# %%
df_train.loc[df_train['Age']< 0, 'Age'] = np.nan

# %%
# 標準化
value_mean = df_train['Fare'].mean()
value_std = df_train['Fare'].std(ddof=0)
# value_std = df_train['Fare'].std() #標本の標準偏差
print('mean:', value_mean, 'std:' , value_std)

# %%
df_train['Fare_standard'] = (df_train['Fare'] - value_mean) / value_std
df_train[['Fare','Fare_standard']].head()
# %%
std = StandardScaler()
std.fit(df_train[['Fare']])
print('mean:', std.mean_[0], ', std:',np.sqrt(std.var_[0]))

df_train['Fare_standard'] = std.transform(df_train[['Fare']])
df_train[['Fare','Fare_standard']].head()

# %% 
# 正規化
value_min = df_train['Fare'].min()
value_max = df_train['Fare'].max()
print('min:', value_min, 'max:' , value_max)
df_train['Fare_normalize'] = (df_train['Fare'] - value_min) / (value_max - value_min)
df_train[['Fare','Fare_normalize']].head()
# %%
mms = MinMaxScaler(feature_range=(0,1))
mms.fit(df_train[['Fare']])
print('min:', mms.data_min_[0], ', max:' , mms.data_max_[0])
df_train['Fare_normalize'] = mms.transform(df_train[['Fare']])
df_train[['Fare','Fare_normalize']].head()

# %% 特徴量生成
# 対数変換 桁が大きい時
df_train['Fare_log'] = np.log(df_train['Fare'] + 1e-5)
df_train[['Fare','Fare_log']].head()

# %% 離散化
df_train['Age_bin'] = pd.cut(
  df_train['Age'],
  bins=[0,10,20,30,40,50,100],
  right=False,
  labels=['10代未満','10代','20代','30代','40代','50代以上'],
  include_lowest=True
)
df_train['Age_bin'] = df_train['Age_bin'].astype(str)
df_train[['Age','Age_bin']].head()

# %%
df_train['Age_na'] = df_train['Age'].isna()*1
df_train[['Age','Age_na']].head(7)

# %% 単変数、カテゴリ変数
ohe_embarled = OneHotEncoder(sparse_output = False) #sparseではエラーになった
ohe_embarled.fit(df_train[['Embarked']])

# for i in ohe_embarled.categories_[0]
tmp_embarked = pd.DataFrame(
  ohe_embarled.transform(df_train[['Embarked']]),
  columns=[f'Embarked_{i}' for i in ohe_embarled.categories_[0]]
)

df_train = pd.concat([df_train, tmp_embarked], axis=1)
df_train[['Embarked','Embarked_C', 'Embarked_Q','Embarked_S','Embarked_nan' ]].head()
# %%
df_ohe = pd.get_dummies(df_train[['Embarked', 'Sex']], dummy_na=True, drop_first=False,dtype='uint8') #dtype指定しないとTrue,Falseになる
df_ohe.head()
# %%
ce_Embarked = df_train['Embarked'].value_counts().to_dict()
print(ce_Embarked)

df_train['Embarked_ce'] = df_train['Embarked'].map(ce_Embarked)
df_train[['Embarked', 'Embarked_ce']].head()
# %%ラベルエンコーディング
le_embarked = LabelEncoder()
le_embarked.fit(df_train['Embarked'])
df_train['Embarked_le'] = le_embarked.transform(df_train['Embarked'])
df_train[['Embarked', 'Embarked_le']].head()

# %%
df_train['Embarked_na'] = df_train['Embarked'].isna()*1
df_train.loc[df_train['Embarked'].isna(), ['Embarked', 'Embarked_na']]
# %% 2変数 数値×数値
df_train['Sibsp_+_Parch'] = df_train['SibSp'] + df_train['Parch']
df_train[['SibSp','Parch','Sibsp_+_Parch']].head()

# %% 2変数 数値×カテゴリ変数
df_agg = df_train.groupby('Sex')['Fare'].agg(['mean']).reset_index()
df_agg.columns = ['Sex', 'mean_Fare_by_Sex']
print('集約テーブル')
display(df_agg)

df_train = pd.merge(df_train, df_agg, on='Sex', how='left')
print('結合後テーブル')
display(df_train[['Sex', 'Fare', 'mean_Fare_by_Sex']].head())



# %%
df_train['mean_Fare_by_Sex'] = df_train.groupby('Sex')['Fare'].transform('mean')
df_train[['Sex', 'Fare', 'mean_Fare_by_Sex']].head()


# %% 2変数 カテゴリ変数×カテゴリ変数
df_tbl = pd.crosstab(df_train['Sex'], df_train['Embarked'])
print('集約テーブル（クロス集計)')
display(df_tbl)
# %%
df_tbl = df_tbl.reset_index()
df_tbl
# %%
df_tbl =pd.melt(df_tbl, id_vars='Sex', value_name='count_Sex_x_Embarked')
print('集約テーブル（縦持ち返還後）')
display(df_tbl)
# %%
df_train = pd.merge(df_train, df_tbl, on=['Sex', 'Embarked'], how='left')
print('結合後テーブル')
df_train[['Sex', 'Embarked', 'count_Sex_x_Embarked']].head()
# %%
df_train['count_Sex_x_Embarked'] = df_train.groupby(['Sex','Embarked'])['PassengerId'].transform('count')
df_train[['Sex', 'Embarked', 'count_Sex_x_Embarked']].head()

# %%
df_sam = df_train.groupby(['Sex','Embarked'])['PassengerId'].count()
df_sam
# %% 2カテ 出現割合
df_tbl = pd.crosstab(df_train['Sex'], df_train['Embarked'], normalize='index')
display(df_tbl)


# %%
df_tbl = df_tbl.reset_index()
df_tbl = pd.melt(df_tbl, id_vars='Sex', value_name='rate_Sex_x_Embarked')
display(df_tbl)
# %%
df_train = pd.merge(df_train, df_tbl, on=['Sex', 'Embarked'], how='left')
df_train[['Sex', 'Embarked', 'rate_Sex_x_Embarked']].head()

# %%
df_train['Sex=male_&_Embarked=S'] = np.where((df_train['Sex']=='male') & (df_train['Embarked']=='S'), 1, 0)
df_train[['Sex', 'Embarked', 'Sex=male_&_Embarked=S']].head()
# %%時系列データ
#ラグ特徴量
df1 = pd.DataFrame({"date":pd.date_range("2021-01-01","2021-01-10"),
                    "weather":["晴れ","晴れ","雨","くもり","くもり","晴れ","雨","晴れ","晴れ","晴れ"],
                    })
df1['weathere_shift1'] = df1['weather'].shift(1)
df1
# %%
df1['weathere_shift1'] = df1['weathere_shift1'].interpolate(method='bfill')
df1

# %%
df2 = pd.DataFrame({"id":["A"]*3 + ["B"]*2 + ["C"]*4,
                    "date":["2021-04-02","2021-04-10","2021-04-25",
                            "2021-04-18","2021-04-19",
                            "2021-04-01","2021-04-04","2021-04-09","2021-04-12",
                            ],
                    "money":[1000,2000,900,4000,1800,900,1200,1100,2900],
                    })
df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')

df2['money_shift1'] = df2.groupby('id')['money'].shift(1)
df2
# %%
df2['date_shift1'] = df2.groupby('id')['date'].shift(1)
df2['days_elapsed'] = df2['date'] - df2['date_shift1']
df2['days_elapsed'] = df2['days_elapsed'].dt.days
df2
# %%
# ウィンドウ特徴量（移動平均）
df3 = pd.DataFrame({"date":pd.date_range("2021-01-01","2021-01-10"),
                    "temperature":[8,10,12,11,9,10,12,7,9,10],
                  })
df3['temperature_window3'] = df3['temperature'].rolling(window=3).mean()
df3
# %%
df4 = pd.DataFrame({"id":["A"]*3 + ["B"]*2 + ["C"]*4,
                    "date":["2021-04-02","2021-04-10","2021-04-25",
                            "2021-04-18","2021-04-19",
                            "2021-04-01","2021-04-04","2021-04-09","2021-04-12",
                          ],
                    "money":[1000,2000,900,4000,1800,900,1200,1100,2900],
                    })
df4['date'] = pd.to_datetime(df4['date'], format='%Y-%m-%d')
df4['money_shift'] = df4.groupby('id')['money'].transform(lambda x:x.rolling(window=2).mean())
df4
# %% 累積特徴量
df5 = pd.DataFrame({"date":pd.date_range("2021-01-01","2021-01-10"),
                    "flag_rain":[0,0,1,0,0,0,1,0,0,0],
                  })
df5['flad_rain_cumsum'] = df5['flag_rain'].cumsum()
df5

# %%
df6 = pd.DataFrame({"id":["A"]*3 + ["B"]*2 + ["C"]*4,
                    "date":["2021-04-02","2021-04-10","2021-04-25",
                            "2021-04-18","2021-04-19",
                            "2021-04-01","2021-04-04","2021-04-09","2021-04-12",
                            ],
                    "money":[1000,2000,900,4000,1800,900,1200,1100,2900],
                    })

df6['date'] = pd.to_datetime(df6['date'], format='%Y-%m-%d')
df6['money_cumsum'] = df6.groupby('id')['money'].cumsum()
df6
# %% テキストデータ
from sklearn.feature_extraction.text import CountVectorizer
vec =CountVectorizer(min_df=20)

vec.fit(df_train['Name'])

df_name = pd.DataFrame(vec.transform(df_train['Name']).toarray(), columns=vec.get_feature_names_out())

print(df_name.shape)
df_name.head()
# %%
#!apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
#!pip install mecab-python3
#!pip install unidic-lite
os.environ['MECABRC'] = '/etc/mecabrc'

import MeCab
# %%
print("サンプルデータ:")
df_text = pd.DataFrame({"text": [
    "今日は雨ですね。天気予報では明日も雨です。",
    "雨なので傘を持って行った方がいいです。",
    "天気予報によると明後日は晴れのようです。",
]})
display(df_text)

print("形態素解析+分かち書き:")
wakati = MeCab.Tagger("-Owakati")
df_text["text_wakati"] = df_text["text"].apply(lambda x: wakati.parse(x).replace("\n",""))
display(df_text)
# %%
print('Bowによるベクトル')
vec = CountVectorizer()
vec.fit(df_text['text_wakati'])
df_text_vec = pd.DataFrame(vec.transform(df_text['text_wakati']).toarray(),columns=vec.get_feature_names_out())

df_text_vec.head()
# %%
