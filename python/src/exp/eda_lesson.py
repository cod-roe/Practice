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
