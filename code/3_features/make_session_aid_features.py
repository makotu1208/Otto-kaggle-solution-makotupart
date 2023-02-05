# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import xgboost as xgb
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
import itertools
from collections import Counter
import cudf, itertools
print('We will use RAPIDS version',cudf.__version__)
cudf.set_option("default_integer_bitwidth", 32)
cudf.set_option("default_float_bitwidth", 32)


# ### ts feature

def session_day_feature(prefix, train, output_path):
    
    # time
    train['date'] = cudf.to_datetime((train.ts + 2*60*60) * 1e9)
    train['dow'] = train.date.dt.dayofweek
    train['day'] = train.date.dt.dayofyear
    train['hour'] = train.date.dt.hour
    
    session_last_hour = train.groupby('session')[['aid', 'day', 'hour', 'dow']].last().reset_index()
    session_last_hour.columns = ['session', 'aid', 'day', 'session_hour_last', 'session_dow_last']
    
    # daily share
    session_day_aid = train[['aid', 'day']].value_counts().reset_index()
    session_day_aid.columns = ['aid', 'day', 'day_n']
    session_day = train[['day']].value_counts().reset_index()
    session_day.columns = ['day', 'total_action']
    session_day_aid = session_day_aid.merge(session_day, on = 'day', how = 'left')
    session_day_aid['daily_aid_share'] = session_day_aid['day_n'] / session_day_aid['total_action']
    session_day_aid['daily_aid_share'] = session_day_aid['daily_aid_share'].astype(np.float32)
    session_day_aid = session_day_aid[['day', 'aid', 'daily_aid_share']]
    #session_day_aid = session_day_aid.to_pandas().fillna(0)
    
    # unique daily_aid_share
    session_day_unique = train.groupby('day')['session'].nunique().reset_index()
    session_aid_day_unique = train.groupby(['aid', 'day'])['session'].nunique().reset_index()
    session_aid_day_unique = session_aid_day_unique.merge(session_day_unique, on = 'day', how = 'left')
    session_aid_day_unique['session_aid_unique_ratio'] = session_aid_day_unique['session_x'] / session_aid_day_unique['session_y']
    session_aid_day_unique['session_aid_unique_ratio'] = session_aid_day_unique['session_aid_unique_ratio'] * 10
    session_aid_day_unique = session_aid_day_unique[['aid', 'day', 'session_aid_unique_ratio']]
    
    # unique daily_aid_cart_share
    session_day_cart_unique = train[train['type'] == 1].groupby('day')['session'].nunique().reset_index()
    session_aid_day_cart_unique = train[train['type'] == 1].groupby(['aid', 'day'])['session'].nunique().reset_index()
    session_aid_day_cart_unique = session_aid_day_cart_unique.merge(session_day_cart_unique, on = 'day', how = 'left')
    session_aid_day_cart_unique['session_aid_cart_unique_ratio'] = session_aid_day_cart_unique['session_x'] / session_aid_day_cart_unique['session_y']
    session_aid_day_cart_unique['session_aid_cart_unique_ratio'] = session_aid_day_cart_unique['session_aid_cart_unique_ratio'] * 10
    session_aid_day_cart_unique = session_aid_day_cart_unique[['aid', 'day', 'session_aid_cart_unique_ratio']]
    
    # unique daily_aid_order_share
    session_day_order_unique = train[train['type'] == 2].groupby('day')['session'].nunique().reset_index()
    session_aid_day_order_unique = train[train['type'] == 2].groupby(['aid', 'day'])['session'].nunique().reset_index()
    session_aid_day_order_unique = session_aid_day_order_unique.merge(session_day_order_unique, on = 'day', how = 'left')
    session_aid_day_order_unique['session_aid_order_unique_ratio'] = session_aid_day_order_unique['session_x'] / session_aid_day_order_unique['session_y']
    session_aid_day_order_unique['session_aid_order_unique_ratio'] = session_aid_day_order_unique['session_aid_order_unique_ratio'] * 10
    session_aid_day_order_unique = session_aid_day_order_unique[['aid', 'day', 'session_aid_order_unique_ratio']]
    
    session_day_aid = session_day_aid.merge(session_aid_day_unique, on = ['aid', 'day'], how = 'outer')
    session_day_aid = session_day_aid.merge(session_aid_day_cart_unique, on = ['aid', 'day'], how = 'outer')
    session_day_aid = session_day_aid.merge(session_aid_day_order_unique, on = ['aid', 'day'], how = 'outer')
    
    session_day_aid = session_day_aid.to_pandas().fillna(0)
    
    for col in ['session_aid_unique_ratio','session_aid_cart_unique_ratio', 'session_aid_order_unique_ratio']:
        session_day_aid[col] = session_day_aid[col].astype(np.float32)
        
    session_day_aid.to_parquet(output_path + prefix + 'session_day_df.parquet')
    del session_day_aid


def session_feature(prefix, train, output_path):
    
    # time
    train['date'] = cudf.to_datetime((train.ts + 2*60*60) * 1e9)
    train['dow'] = train.date.dt.dayofweek
    train['day'] = train.date.dt.dayofyear
    train['hour'] = train.date.dt.hour
    
    session_last_hour = train.groupby('session')[['aid', 'day', 'hour', 'dow']].last().reset_index()
    session_last_hour.columns = ['session', 'aid', 'day', 'session_hour_last', 'session_dow_last']
    
    session_last_type = train.groupby('session').last().reset_index()[['session', 'type']]
    session_last_type.columns = ['session', 'last_type']
    
    # all
    session_df = train['session'].value_counts().reset_index()
    session_df.columns = ['session', 'all_counts']

    click_num =  train[train['type'] == 0][['session']].value_counts().reset_index().rename(columns={0: 'click_counts'})
    cart_num =  train[train['type'] == 1][['session']].value_counts().reset_index().rename(columns={0: 'cart_counts'})
    order_num =  train[train['type'] == 2][['session']].value_counts().reset_index().rename(columns={0: 'order_counts'})

    session_df = session_df.merge(click_num, on = 'session', how = 'left')
    session_df = session_df.merge(cart_num, on = 'session', how = 'left')
    session_df = session_df.merge(order_num, on = 'session', how = 'left')
    session_df['click_ratio'] = session_df['click_counts'] / session_df['all_counts']
    session_df['cart_ratio'] = session_df['cart_counts'] / session_df['all_counts']
    session_df['order_ratio'] = session_df['order_counts'] / session_df['all_counts']
    session_df['session_cart_cvr'] = session_df['cart_counts'] / session_df['click_counts']
    session_df['session_order_cvr'] = session_df['order_counts'] / session_df['cart_counts']
    session_df = session_df[['session', 'all_counts', 'click_ratio', 'cart_ratio', 'order_ratio', 
                             'session_cart_cvr', 'session_order_cvr']]

    del click_num, cart_num, order_num
    
    train_max_ts = train.groupby('session')['ts'].max().reset_index()
    train_max_ts.columns = ['session', 'max_ts']
    train_max_ts['day_ts'] = train_max_ts['max_ts'] - (24 * 60 * 60)
    
    # last 1day action
    train = train.merge(train_max_ts, on = 'session', how = 'left')
    train_last_day = train[train['ts'] >= train['day_ts']]
    
    # all
    session_lastday_df = train_last_day['session'].value_counts().reset_index()
    session_lastday_df.columns = ['session', 'lastday_all_counts']

    click_num =  train_last_day[train_last_day['type'] == 0][['session']].value_counts().reset_index().rename(columns={0: 'lastday_click_counts'})
    cart_num =  train_last_day[train_last_day['type'] == 1][['session']].value_counts().reset_index().rename(columns={0: 'lastday_cart_counts'})
    order_num =  train_last_day[train_last_day['type'] == 2][['session']].value_counts().reset_index().rename(columns={0: 'lastday_order_counts'})

    session_lastday_df = session_lastday_df.merge(click_num, on = 'session', how = 'left')
    session_lastday_df = session_lastday_df.merge(cart_num, on = 'session', how = 'left')
    session_lastday_df = session_lastday_df.merge(order_num, on = 'session', how = 'left')
    session_lastday_df['lastday_click_ratio'] = session_lastday_df['lastday_click_counts'] / session_lastday_df['lastday_all_counts']
    session_lastday_df['lastday_cart_ratio'] = session_lastday_df['lastday_cart_counts'] / session_lastday_df['lastday_all_counts']
    session_lastday_df['lastday_order_ratio'] = session_lastday_df['lastday_order_counts'] / session_lastday_df['lastday_all_counts']
    session_lastday_df['lastday_session_cart_cvr'] = session_lastday_df['lastday_cart_counts'] / session_lastday_df['lastday_click_counts']
    session_lastday_df['lastday_session_order_cvr'] = session_lastday_df['lastday_order_counts'] / session_lastday_df['lastday_cart_counts']
    session_lastday_df = session_lastday_df[['session', 'lastday_all_counts', 'lastday_click_ratio', 
                                             'lastday_cart_ratio', 'lastday_order_ratio', 
                                             'lastday_session_cart_cvr', 'lastday_session_order_cvr']]

    del click_num, cart_num, order_num
    
    session_nunique_df = train.groupby(['session'])['aid'].nunique().reset_index()
    session_nunique_df.columns = ['session', 'nunique_aids']
    
    session_ts_length = train.groupby('session').agg({'ts': ['min', 'max']}).reset_index()
    session_ts_length.columns = ['session', 'ts_min', 'ts_max']
    session_ts_length['ts_length'] = (session_ts_length['ts_max'] - session_ts_length['ts_min']) / 3600
    session_ts_length = session_ts_length[['session', 'ts_length']]
    
    session_ts_nunique = train.groupby('session')['ts'].nunique().reset_index()
    session_ts_nunique.columns = ['session', 'ts_nunique']
    
    session_ts_unique = train[['session', 'ts']].drop_duplicates()
    session_ts_unique['ts_diff'] = session_ts_unique.groupby('session')['ts'].diff()
    
    session_last_diff = (session_ts_unique.groupby('session')['ts_diff'].last() / 3600).reset_index()
    session_last_diff.columns = ['session', 'session_last_diff']
    
    session_ts_unique = session_ts_unique.groupby('session')['ts_diff'].mean().reset_index()
    session_ts_unique['ts_diff_mean'] = cudf.DataFrame(np.log1p(session_ts_unique.to_pandas().fillna(0)['ts_diff']))
    session_ts_unique = session_ts_unique[['session', 'ts_diff_mean']]
    
    # join
    session_df = session_df.merge(session_lastday_df, on = 'session', how = 'left')
    session_df = session_df.merge(session_nunique_df, on = 'session', how = 'left')
    session_df = session_df.merge(session_ts_length, on = 'session', how = 'left')
    session_df = session_df.merge(session_ts_nunique, on = 'session', how = 'left')
    session_df = session_df.merge(session_last_diff, on = 'session', how = 'left')
    
    session_df['count_per_ts'] = session_df['ts_length'] / session_df['all_counts']
    session_df['count_per_aids'] = session_df['nunique_aids'] / session_df['all_counts']
    session_df['ts_per_length'] = session_df['ts_length'] / session_df['ts_nunique']
    
    # join session ts
    session_df = session_df.merge(session_last_hour, on = 'session', how = 'left')
    session_df = session_df.merge(session_last_type, on = 'session', how = 'left')
    session_df = session_df.to_pandas().fillna(0)
    
    float32_cols = ['click_ratio', 'cart_ratio', 'order_ratio', 'session_cart_cvr', 'session_order_cvr',
                    'lastday_click_ratio', 'lastday_cart_ratio', 'lastday_order_ratio', 
                    'lastday_session_cart_cvr', 'lastday_session_order_cvr',
                    'count_per_ts', 'count_per_aids', 'ts_per_length']

    for col in float32_cols:
        session_df[col] = session_df[col].astype(np.float32)
        
    session_df = session_df[['session', 'aid', 'day', 'last_type', 'all_counts', 'click_ratio', 
                             'cart_ratio', 'order_ratio', 'session_cart_cvr', 'session_order_cvr',
                             'nunique_aids', 'count_per_ts', 'count_per_aids', 'ts_per_length', 
                             'session_hour_last', 'session_dow_last']]
    
    session_df.to_parquet(output_path + prefix + 'session_df.parquet')
    del session_df, session_nunique_df, session_ts_length, session_ts_nunique, session_last_diff


def aid_feature(prefix, merge, train, output_path):
    
    # base_df
    aid_features = merge[['aid']].drop_duplicates().reset_index(drop=True)
    
    chunk = 7000
    co_matrix = []
    cart_cvr_df = []
    chunk_num = int(len(merge['aid'].drop_duplicates()) / chunk) + 1

    for i in tqdm(range(chunk_num)):

        start = i * chunk
        end = (i + 1) * chunk

        row = merge[(merge['aid'] >= start) & (merge['aid'] < end) & (merge['type'] == 0)]
        row_cart = row.merge(merge[merge['type'] == 1], on = 'session', how = 'inner')
        row_order = row.merge(merge[merge['type'] == 2], on = 'session', how = 'inner')

        click_all = row[['aid', 'session']].drop_duplicates()['aid'].value_counts().reset_index()
        click_cart = row_cart[['session', 'aid_x']].drop_duplicates()['aid_x'].value_counts().reset_index()
        click_order = row_order[['session', 'aid_x']].drop_duplicates()['aid_x'].value_counts().reset_index()

        click_all.columns = ['aid', 'click_n']
        click_cart.columns = ['aid', 'cart_n']
        click_order.columns = ['aid', 'order_n']
        click_all = click_all.merge(click_cart, on = 'aid', how = 'left')
        click_all = click_all.merge(click_order, on = 'aid', how = 'left')
        click_all['cart_hist_cvr'] = click_all['cart_n'] / click_all['click_n'] 
        click_all['order_hist_cvr'] = click_all['order_n'] / click_all['click_n'] 

        cart_cvr_df.append(click_all)

    cart_cvr_df = cudf.concat(cart_cvr_df)
    cart_cvr_df = cart_cvr_df[['aid', 'cart_hist_cvr', 'order_hist_cvr']]

    del click_all, click_cart, click_order, row
    
    # cart : skip click ratio
    cart_df = merge[merge['type'] == 1][['session', 'aid']].drop_duplicates()['aid'].value_counts().reset_index()
    cart_df.columns = ['aid', 'cart_unique_num']

    cart_session = list(merge[merge['type'] == 1]['session'].unique().to_pandas())
    merge_cart = merge[merge['session'].isin(cart_session)]
    merge_pivot = merge_cart.pivot_table(index=['session', 'aid'], columns=['type'], aggfunc='count').reset_index()
    merge_pivot.columns = ['session', 'aid', 'count_0', 'count_1', 'count_2']
    merge_pivot = cudf.DataFrame(merge_pivot.to_pandas().fillna(0))
    
    true_click_cvr = merge_pivot[merge_pivot['count_0'] > 0].groupby('aid').agg({'count_0':'sum', 
                                                                                 'count_1':'sum',
                                                                                 'count_2':'sum',
                                                                                }).reset_index()
    true_click_cvr['true_cart_cvr'] = true_click_cvr['count_1'] / true_click_cvr['count_0']
    true_click_cvr['true_order_cvr'] = true_click_cvr['count_2'] / true_click_cvr['count_1']
    true_click_cvr = true_click_cvr[['aid', 'true_cart_cvr', 'true_order_cvr']]

    del merge_pivot
    
    all_aid_list = list(merge['aid'].unique().to_pandas())
    
    merge['before_ts'] = merge.groupby(['session', 'aid'])['ts'].shift()
    merge['before_type'] = merge.groupby(['session', 'aid'])['type'].shift()
    merge_ts = merge[merge['before_type'] >= 0]
    merge_ts['type_cvr'] = merge_ts['type'].astype('str') + '_' + merge_ts['before_type'].astype('str')
    merge_ts['diff_ts'] = (merge_ts['ts'] - merge_ts['before_ts']) / 3600
    
    type_counts = merge_ts[['aid', 'type_cvr']].value_counts().reset_index()
    type_counts.columns = ['aid', 'type_cvr', 'n']
    type_counts = type_counts.pivot_table(index = ['aid'], columns = ['type_cvr'], values = ['n']).reset_index()
    type_counts.columns = ['aid', 
                           'aid_click_click_ratio', 'aid_cart_click_ratio', 'aid_order_click_ratio',
                           'aid_click_cart_ratio',  'aid_cart_cart_ratio',  'aid_order_cart_ratio',
                           'aid_click_order_ratio', 'aid_cart_order_ratio', 'aid_order_order_ratio']
    type_counts['cvr_sum'] = type_counts.iloc[:,1:].sum(1)

    for i in list(type_counts.columns)[1:10]:
        type_counts[i] = type_counts[i] / type_counts['cvr_sum']

    type_counts = cudf.DataFrame(type_counts.to_pandas().fillna(0))
    type_counts['cvr_sum_share'] = type_counts['cvr_sum'] / type_counts['cvr_sum'].sum()
    
    type_ts_diff = merge_ts.groupby(['aid', 'type_cvr'])['diff_ts'].mean().reset_index()
    type_ts_diff = type_ts_diff.pivot_table(index = ['aid'], columns = ['type_cvr'], values = ['diff_ts']).reset_index()
    type_ts_diff.columns = ['aid', 
                           'aid_click_click_diffts', 'aid_cart_click_diffts', 'aid_order_click_diffts',
                           'aid_click_cart_diffts',  'aid_cart_cart_diffts',  'aid_order_cart_diffts',
                           'aid_click_order_diffts', 'aid_cart_order_diffts', 'aid_order_order_diffts']
    type_ts_diff = type_ts_diff[['aid', 'aid_click_click_diffts', 'aid_click_cart_diffts', 
                                 'aid_cart_click_diffts', 'aid_cart_order_diffts']]
    
    del merge['before_ts'], merge['before_type']
    
    # cart : skip click ratio
    cart_df = merge[merge['type'] == 1][['session', 'aid']].drop_duplicates()['aid'].value_counts().reset_index()
    cart_df.columns = ['aid', 'cart_unique_num']

    cart_session = list(merge[merge['type'] == 1]['session'].unique().to_pandas())
    merge_cart = merge[merge['session'].isin(cart_session)]
    merge_pivot = merge_cart.pivot_table(index=['session', 'aid'], columns=['type'], aggfunc='count').reset_index()
    merge_pivot.columns = ['session', 'aid', 'count_0', 'count_1', 'count_2']
    merge_pivot = cudf.DataFrame(merge_pivot.to_pandas().fillna(0))

    # cart - click ratio
    skip_aid_df = merge_pivot[(merge_pivot['count_1'] -  merge_pivot['count_0']) > 0][['aid']].value_counts().reset_index()
    skip_aid_df.columns = ['aid', 'cart_click_skip_num']

    cart_df = cart_df.merge(skip_aid_df, on = ['aid'], how = 'left')
    cart_df = cudf.DataFrame(cart_df.to_pandas().fillna(0))
    cart_df['cart_click_skip_ratio'] = cart_df['cart_click_skip_num'] / cart_df['cart_unique_num']
    cart_df = cart_df[['aid', 'cart_click_skip_ratio']]

    del merge_pivot
    
    # order : skip click ratio / skip cart ratio
    order_df = merge[merge['type'] == 2][['session', 'aid']].drop_duplicates()['aid'].value_counts().reset_index()
    order_df.columns = ['aid', 'order_unique_num']

    order_session = list(merge[merge['type'] == 2]['session'].unique().to_pandas())
    merge_order = merge[merge['session'].isin(order_session)]
    merge_pivot = merge_order.pivot_table(index=['session', 'aid'], columns=['type'], aggfunc='count').reset_index()
    merge_pivot.columns = ['session', 'aid', 'count_0', 'count_1', 'count_2']
    merge_pivot = cudf.DataFrame(merge_pivot.to_pandas().fillna(0))

    # order - click ratio
    skip_aid_df_1 = merge_pivot[(merge_pivot['count_2'] -  merge_pivot['count_0']) > 0][['aid']].value_counts().reset_index()
    skip_aid_df_1.columns = ['aid', 'order_click_skip_num']

    # order - cart ratio
    skip_aid_df_2 = merge_pivot[(merge_pivot['count_2'] -  merge_pivot['count_1']) > 0][['aid']].value_counts().reset_index()
    skip_aid_df_2.columns = ['aid', 'order_cart_skip_num']

    order_df = order_df.merge(skip_aid_df_1, on = ['aid'], how = 'left')
    order_df = order_df.merge(skip_aid_df_2, on = ['aid'], how = 'left')
    order_df = cudf.DataFrame(order_df.to_pandas().fillna(0))
    order_df['order_click_skip_ratio'] = order_df['order_click_skip_num'] / order_df['order_unique_num']
    order_df['order_cart_skip_ratio'] = order_df['order_cart_skip_num'] / order_df['order_unique_num']
    order_df = order_df[['aid', 'order_cart_skip_ratio', 'order_click_skip_ratio']]

    del merge_pivot
    
    # base_df
    aid_features = merge[['aid']].drop_duplicates().reset_index(drop=True)
    
    unique_click_session = merge[merge['type'] == 0].groupby(['aid'])['session'].nunique().reset_index()
    unique_click_session.columns = ['aid', 'click_session']

    unique_cart_session = merge[merge['type'] == 1].groupby(['aid'])['session'].nunique().reset_index()
    unique_cart_session.columns = ['aid', 'cart_session']

    unique_order_session = merge[merge['type'] == 2].groupby(['aid'])['session'].nunique().reset_index()
    unique_order_session.columns = ['aid', 'order_session']
    
    unique_click_session = merge[merge['type'] == 0].groupby(['aid'])['session'].nunique().reset_index()
    unique_click_session.columns = ['aid', 'click_session']

    unique_cart_session = merge[merge['type'] == 1].groupby(['aid'])['session'].nunique().reset_index()
    unique_cart_session.columns = ['aid', 'cart_session']

    unique_order_session = merge[merge['type'] == 2].groupby(['aid'])['session'].nunique().reset_index()
    unique_order_session.columns = ['aid', 'order_session']
    
    unique_session = unique_click_session.merge(unique_cart_session, on = 'aid', how = 'left')
    unique_session = unique_session.merge(unique_order_session, on = 'aid', how = 'left')
    unique_session['click_cvr_unique'] = unique_session['click_session'] / len(merge['aid'].unique())
    unique_session['cart_cvr_unique'] = unique_session['cart_session'] / unique_session['click_session']
    unique_session['order_cvr_unique'] = unique_session['order_session'] / unique_session['cart_session']
    unique_session['click_order_cvr_unique'] = unique_session['order_session'] / unique_session['click_session']
    #unique_session = unique_session[['aid', 'click_cvr_unique', 'cart_cvr_unique', 'order_cvr_unique']]
    unique_session['click_cvr_unique'] = np.log(unique_session['click_cvr_unique'].to_pandas())

    # cvr_df(unique user)
    click_counts = merge[merge['type'] == 0]['aid'].value_counts().reset_index()
    click_counts.columns = ['aid', 'click_n']
    cart_counts = merge[merge['type'] == 1]['aid'].value_counts().reset_index()
    cart_counts.columns = ['aid', 'cart_n']
    order_counts = merge[merge['type'] == 2]['aid'].value_counts().reset_index()
    order_counts.columns = ['aid', 'order_n']
    
    # cvr_df(all)
    click_counts = merge[merge['type'] == 0]['aid'].value_counts().reset_index()
    click_counts.columns = ['aid', 'click_n']
    cart_counts = merge[merge['type'] == 1]['aid'].value_counts().reset_index()
    cart_counts.columns = ['aid', 'cart_n']
    order_counts = merge[merge['type'] == 2]['aid'].value_counts().reset_index()
    order_counts.columns = ['aid', 'order_n']

    cvr_df = click_counts.merge(cart_counts, on = 'aid', how = 'left')
    cvr_df = cvr_df.merge(order_counts, on = 'aid', how = 'left')

    cvr_df['cart_cvr'] = cvr_df['cart_n'] / cvr_df['click_n']
    cvr_df['order_cvr'] = cvr_df['order_n'] / cvr_df['cart_n']
    cvr_df['click_order_cvr'] = cvr_df['order_n'] / cvr_df['click_n']
    #cvr_df = cvr_df[['aid', 'cart_cvr', 'order_cvr']]

    cvr_df['cart_cvr'] = cvr_df['cart_cvr'].astype(np.float32)
    cvr_df['order_cvr'] = cvr_df['order_cvr'].astype(np.float32)
    cvr_df['click_order_cvr'] = cvr_df['click_order_cvr'].astype(np.float32)
    
    repeat_df = unique_session[['aid', 'click_session', 'cart_session', 'order_session']].merge(
        cvr_df[['aid', 'click_n', 'cart_n', 'order_n']], on = 'aid', how = 'left')

    repeat_df['click_repeat_ratio'] = repeat_df['click_session'] / repeat_df['click_n']
    repeat_df['cart_repeat_ratio'] = repeat_df['cart_session'] / repeat_df['cart_n']
    repeat_df['order_repeat_ratio'] = repeat_df['order_session'] / repeat_df['order_n']
    repeat_df = repeat_df[['aid', 'click_repeat_ratio', 'cart_repeat_ratio', 'order_repeat_ratio']]
    
    # cvr_df(test)
    click_counts = train[train['type'] == 0]['aid'].value_counts().reset_index()
    click_counts.columns = ['aid', 'click_n']
    cart_counts = train[train['type'] == 1]['aid'].value_counts().reset_index()
    cart_counts.columns = ['aid', 'cart_n']
    order_counts = train[train['type'] == 2]['aid'].value_counts().reset_index()
    order_counts.columns = ['aid', 'order_n']

    cvr_df_test = click_counts.merge(cart_counts, on = 'aid', how = 'left')
    cvr_df_test = cvr_df_test.merge(order_counts, on = 'aid', how = 'left')

    cvr_df_test['cart_cvr_test'] = cvr_df_test['cart_n'] / cvr_df_test['click_n']
    cvr_df_test['order_cvr_test'] = cvr_df_test['order_n'] / cvr_df_test['cart_n']
    cvr_df_test = cvr_df_test[['aid', 'cart_cvr_test', 'order_cvr_test']]

    cvr_df_test['cart_cvr_test'] = cvr_df_test['cart_cvr_test'].astype(np.float32)
    cvr_df_test['order_cvr_test'] = cvr_df_test['order_cvr_test'].astype(np.float32)
    
    # all term share
    click_share_all = merge[merge['type'] == 0]['aid'].value_counts(normalize = True).reset_index()
    click_share_all.columns = ['aid', 'click_share_all']
    cart_share_all = merge[merge['type'] == 1]['aid'].value_counts(normalize = True).reset_index()
    cart_share_all.columns = ['aid', 'cart_share_all']
    order_share_all = merge[merge['type'] == 2]['aid'].value_counts(normalize = True).reset_index()
    order_share_all.columns = ['aid', 'order_share_all']

    # test term share
    click_share_test = train[train['type'] == 0]['aid'].value_counts(normalize = True).reset_index()
    click_share_test.columns = ['aid', 'click_share_test']
    cart_share_test = train[train['type'] == 1]['aid'].value_counts(normalize = True).reset_index()
    cart_share_test.columns = ['aid', 'cart_share_test']
    order_share_test = train[train['type'] == 2]['aid'].value_counts(normalize = True).reset_index()
    order_share_test.columns = ['aid', 'order_share_test']
    
    session_aid_ts = merge[['session', 'aid', 'ts']].sort_values(['session', 'ts'])
    session_aid_ts['aid_next_diff_mean'] = session_aid_ts.groupby(['session', 'aid'])['ts'].diff()
    session_diff_action = session_aid_ts.groupby(['aid'])['aid_next_diff_mean'].mean().reset_index()
    
    merge_ts = merge.groupby(['session', 'aid'])['ts'].first().reset_index()
    merge_ts = merge_ts.sort_values(['session', 'ts']).reset_index(drop=True)
    merge_ts['rank'] = 1
    merge_ts['rank'] = merge_ts.groupby('session')['rank'].cumsum()
    merge_ts = merge_ts.groupby('aid')['rank'].mean().reset_index()
    merge_ts.columns = ['aid', 'aid_rank_mean']
    
    # aidの最終action ts
    aid_max_df = merge.groupby('aid')['ts'].max().reset_index()
    aid_max_df['aid_last_action_diff'] = (merge['ts'].max() - aid_max_df['ts'] ) / 3600
    aid_max_df = aid_max_df[['aid', 'aid_last_action_diff']]
    
    # merge
    aid_features = aid_features.merge(cart_cvr_df, on = 'aid', how = 'left')
    aid_features = aid_features.merge(cvr_df, on = 'aid', how = 'left')
    aid_features = aid_features.merge(cvr_df_test, on = 'aid', how = 'left')
    aid_features = aid_features.merge(click_share_all, on = 'aid', how = 'left')
    aid_features = aid_features.merge(click_share_test, on = 'aid', how = 'left')
    aid_features = aid_features.merge(cart_share_all, on = 'aid', how = 'left')
    aid_features = aid_features.merge(cart_share_test, on = 'aid', how = 'left')
    aid_features = aid_features.merge(order_share_all, on = 'aid', how = 'left')
    aid_features = aid_features.merge(order_share_test, on = 'aid', how = 'left')
    aid_features = aid_features.merge(session_diff_action, on = 'aid', how = 'left')
    aid_features = aid_features.merge(unique_session, on = 'aid', how = 'left')
    aid_features = aid_features.merge(repeat_df, on = 'aid', how = 'left')
    aid_features = aid_features.merge(merge_ts, on = 'aid', how = 'left')
    aid_features = aid_features.merge(aid_max_df, on = 'aid', how = 'left')
    aid_features = aid_features.merge(cart_df, on = 'aid', how = 'left')
    aid_features = aid_features.merge(order_df, on = 'aid', how = 'left')

    aid_features = aid_features.merge(type_counts, on = 'aid', how = 'left')
    aid_features = aid_features.merge(type_ts_diff, on = 'aid', how = 'left')
    
    del cvr_df, cvr_df_test, click_share_all, click_share_test, cart_share_all, cart_share_test, order_share_all, order_share_test
    del session_diff_action, unique_session, repeat_df, aid_max_df
    del cart_df, order_df
    del type_counts, type_ts_diff
    
    float32_cols = ['cart_hist_cvr', 'order_hist_cvr', 'aid_next_diff_mean', 'click_cvr_unique', 
                    'cart_cvr_unique', 'order_cvr_unique', 'click_order_cvr_unique', 
                    'click_repeat_ratio', 'cart_repeat_ratio', 'order_repeat_ratio', 'aid_rank_mean',
                   'cart_click_skip_ratio', 'order_cart_skip_ratio', 'order_click_skip_ratio',
                   'aid_click_click_ratio', 'aid_cart_click_ratio',
                   'aid_order_click_ratio', 'aid_click_cart_ratio', 'aid_cart_cart_ratio',
                   'aid_order_cart_ratio', 'aid_click_order_ratio', 'aid_cart_order_ratio',
                   'aid_order_order_ratio', 'cvr_sum', 'cvr_sum_share',
                   'aid_click_click_diffts', 'aid_click_cart_diffts',
                   'aid_cart_click_diffts', 'aid_cart_order_diffts']

    for col in float32_cols:
        aid_features[col] = aid_features[col].astype(np.float32)
        
    aid_features = aid_features.to_pandas().fillna(0)
    aid_features['aid_next_diff_mean'] = np.log1p(aid_features['aid_next_diff_mean'])
    
    aid_features = aid_features[['aid', 'cart_hist_cvr', 'order_hist_cvr', 'cart_cvr', 'order_cvr', 
                                 'click_order_cvr', 'cart_cvr_test', 'order_cvr_test',
                                 'click_cvr_unique', 'cart_cvr_unique', 'order_cvr_unique', 'click_order_cvr_unique',
                                 'click_share_all', 'click_share_test', 'cart_share_all',
                                 'cart_share_test', 'order_share_all', 'order_share_test', 'aid_next_diff_mean',
                                'click_repeat_ratio', 'cart_repeat_ratio', 'order_repeat_ratio', 'aid_rank_mean',
                                'aid_last_action_diff', 'cart_click_skip_ratio', 'order_cart_skip_ratio', 
                                 'order_click_skip_ratio', 'aid_click_click_ratio', 'aid_cart_click_ratio',
                               'aid_order_click_ratio', 'aid_click_cart_ratio', 'aid_cart_cart_ratio',
                               'aid_order_cart_ratio', 'aid_click_order_ratio', 'aid_cart_order_ratio',
                               'aid_order_order_ratio', 'cvr_sum', 'cvr_sum_share',
                               'aid_click_click_diffts', 'aid_click_cart_diffts',
                               'aid_cart_click_diffts', 'aid_cart_order_diffts']]
    
    aid_features.to_parquet(output_path + prefix + 'aid_features_df.parquet')
    del aid_features


# ### last-chunk session-aid

def last_chunk_session_aid(prefix, train, output_path):
    
    train_chunk = train.copy()
    train_chunk['ts_diff'] = train_chunk.groupby('session')['ts'].diff()
    train_chunk['ts_diff'] = train_chunk.to_pandas()['ts_diff'].fillna(0)
    # hour chunk
    train_chunk['chunk_flg'] = np.where(train_chunk.to_pandas().ts_diff < 3600, 0, 1)
    train_chunk['chunk'] = train_chunk.groupby('session')['chunk_flg'].cumsum()

    train_max_chunk = train_chunk.groupby('session')['chunk'].max().reset_index()
    train_max_chunk.columns = ['session', 'max_chunk']
    train_chunk = train_chunk.merge(train_max_chunk, on = 'session', how = 'left')
    train_chunk = train_chunk.sort_values(['session', 'ts']).reset_index(drop=True)
    train_chunk['last_chunk_num'] = train_chunk.groupby(['session', 'chunk'])['ts'].rank(ascending = False, method = 'max')
    
    chunk_count = train_chunk[['session', 'chunk']].value_counts().reset_index()
    chunk_count.columns = ['session', 'chunk', 'chunk_counts']
    chunk_count = chunk_count.groupby('session')['chunk_counts'].mean().reset_index()
    chunk_count.columns = ['session', 'session_counts_mean']
    
    train_last_chunk = train_chunk[train_chunk['chunk'] == train_chunk['max_chunk']]
    train_last_chunk = train_last_chunk.pivot_table(index = ['session', 'aid'], 
                                                    columns = ['type'], 
                                                    values = ['ts'], 
                                                    aggfunc='count').reset_index()
    train_last_chunk.columns = ['session', 'aid', 'last_chunk_click', 'last_chunk_cart', 'last_chunk_order']
    train_last_chunk = cudf.DataFrame(train_last_chunk.to_pandas().fillna(0))
    train_last_chunk['last_chunk_aid_total'] = train_last_chunk.iloc[:,2:].sum(1)

    train_last_chunk = train_last_chunk.merge(train_max_chunk, on = 'session', how = 'left')
    train_last_chunk = train_last_chunk.merge(chunk_count, on = 'session', how = 'left')

    train_last_chunk_total = train_last_chunk.groupby('session')['last_chunk_aid_total'].sum().reset_index()
    train_last_chunk_total.columns = ['session', 'last_chunk_total']
    train_last_chunk = train_last_chunk.merge(train_last_chunk_total, on = 'session', how = 'left')
    train_last_chunk['last_chunk_aid_ratio'] = train_last_chunk['last_chunk_aid_total'] / train_last_chunk['last_chunk_total']
    train_chunk_last_num = train_chunk.groupby(['session', 'aid'])['last_chunk_num'].min().reset_index()
    train_last_chunk = train_last_chunk.merge(train_chunk_last_num, on = ['session', 'aid'], how = 'left')
    
    float32_cols = ['last_chunk_click', 'last_chunk_cart', 'last_chunk_order', 'last_chunk_aid_total', 
                    'session_counts_mean', 'last_chunk_total', 'last_chunk_aid_ratio']
    int32_cols = ['max_chunk', 'last_chunk_num']

    for col in float32_cols:
        train_last_chunk[col] = train_last_chunk[col].astype(np.float32)

    for col in int32_cols:
        train_last_chunk[col] = train_last_chunk[col].astype(np.int32)
        
    train_last_chunk.to_parquet(output_path + prefix + 'last_chunk_session_aid_df.parquet')
    del train_last_chunk


def user_aid_feature(prefix, train, output_path):
    
    session_aid_total_df = train[['session']].value_counts().reset_index()
    session_aid_total_df.columns = ['session', 'session_total_action']
    
    session_aid_df = train[['session', 'aid']].value_counts().reset_index()
    session_aid_df.columns = ['session', 'aid', 'session_aid_total_action']
    session_aid_df = session_aid_df.merge(session_aid_total_df, on = 'session', how = 'left')
    session_aid_df['session_aid_share'] = session_aid_df['session_aid_total_action'] / session_aid_df['session_total_action']
    
    session_aid_click_df = train[train['type'] == 0][['session', 'aid']].value_counts().reset_index()
    session_aid_click_df.columns = ['session', 'aid', 'session_aid_total_click']
    session_aid_df = session_aid_df.merge(session_aid_click_df, on = ['session', 'aid'], how = 'left')
    session_aid_df['session_aid_click_share'] = session_aid_df['session_aid_total_click'] / session_aid_df['session_aid_total_action']
    
    session_aid_cart_df = train[train['type'] == 1][['session', 'aid']].value_counts().reset_index()
    session_aid_cart_df.columns = ['session', 'aid', 'session_aid_total_cart']
    session_aid_df = session_aid_df.merge(session_aid_cart_df, on = ['session', 'aid'], how = 'left')
    session_aid_df['session_aid_cart_share'] = session_aid_df['session_aid_total_cart'] / session_aid_df['session_aid_total_action']
    
    session_aid_order_df = train[train['type'] == 2][['session', 'aid']].value_counts().reset_index()
    session_aid_order_df.columns = ['session', 'aid', 'session_aid_total_order']
    session_aid_df = session_aid_df.merge(session_aid_order_df, on = ['session', 'aid'], how = 'left')
    session_aid_df['session_aid_order_share'] = session_aid_df['session_aid_total_order'] / session_aid_df['session_aid_total_action']
    
    session_aid_df = session_aid_df[['session', 'aid', 
                                     'session_aid_total_action', 'session_aid_share',
                                     'session_aid_click_share', 'session_aid_cart_share', 'session_aid_order_share'
                                    ]]
    del session_aid_click_df, session_aid_cart_df, session_aid_order_df
    
    float32_cols = ['session_aid_total_action', 'session_aid_share',
                    'session_aid_click_share', 'session_aid_cart_share', 'session_aid_order_share']

    for col in float32_cols:
        session_aid_df[col] = session_aid_df[col].astype(np.float32)
        
    # last 1hour click
    train_last_ts = train.groupby(['session'])['ts'].max().reset_index()
    train_last_ts['diff_1hour']  = train_last_ts['ts'] - (60 * 60)
    train_last_ts.columns = ['session', 'ts_max', 'diff_1hour']
    train_last_ts = train.merge(train_last_ts[['session', 'diff_1hour']], on ='session', how = 'left')
    train_last_ts = train_last_ts[train_last_ts['ts'] >= train_last_ts['diff_1hour']]

    train_last_1hour_click = train_last_ts[train_last_ts['type'] == 0][['session', 'aid']].value_counts().reset_index()
    train_last_1hour_click.columns = ['session', 'aid', 'last_1hour_clicks']

    train_last_1hour_cart = train_last_ts[train_last_ts['type'] == 1][['session', 'aid']].value_counts().reset_index()
    train_last_1hour_cart.columns = ['session', 'aid', 'last_1hour_carts']

    train_last_1hour_order = train_last_ts[train_last_ts['type'] == 2][['session', 'aid']].value_counts().reset_index()
    train_last_1hour_order.columns = ['session', 'aid', 'last_1hour_orders']
    
    del train_last_ts
    
    # last day click
    train_last_ts = train.groupby(['session'])['ts'].max().reset_index()
    train_last_ts['diff_1day']  = train_last_ts['ts'] - (24 * 60 * 60)
    train_last_ts.columns = ['session', 'ts_max', 'diff_1day']
    train_last_ts = train.merge(train_last_ts[['session', 'diff_1day']], on ='session', how = 'left')
    train_last_ts = train_last_ts[train_last_ts['ts'] >= train_last_ts['diff_1day']]

    train_last_1day_click = train_last_ts[train_last_ts['type'] == 0][['session', 'aid']].value_counts().reset_index()
    train_last_1day_click.columns = ['session', 'aid', 'last_1day_clicks']

    train_last_1day_cart = train_last_ts[train_last_ts['type'] == 1][['session', 'aid']].value_counts().reset_index()
    train_last_1day_cart.columns = ['session', 'aid', 'last_1day_carts']

    train_last_1day_order = train_last_ts[train_last_ts['type'] == 2][['session', 'aid']].value_counts().reset_index()
    train_last_1day_order.columns = ['session', 'aid', 'last_1day_orders']
    del train_last_ts
    
    # last week click
    train_last_ts = train.groupby(['session'])['ts'].max().reset_index()
    train_last_ts['diff_1week']  = train_last_ts['ts'] - (7 * 24 * 60 * 60)
    train_last_ts.columns = ['session', 'ts_max', 'diff_1week']
    train_last_ts = train.merge(train_last_ts[['session', 'diff_1week']], on ='session', how = 'left')
    train_last_ts = train_last_ts[train_last_ts['ts'] >= train_last_ts['diff_1week']]

    train_last_1week_click = train_last_ts[train_last_ts['type'] == 0][['session', 'aid']].value_counts().reset_index()
    train_last_1week_click.columns = ['session', 'aid', 'last_1week_clicks']

    train_last_1week_cart = train_last_ts[train_last_ts['type'] == 1][['session', 'aid']].value_counts().reset_index()
    train_last_1week_cart.columns = ['session', 'aid', 'last_1week_carts']

    train_last_1week_order = train_last_ts[train_last_ts['type'] == 2][['session', 'aid']].value_counts().reset_index()
    train_last_1week_order.columns = ['session', 'aid', 'last_1week_orders']
    
    # 各itemが最後にいつactionされたか
    train_last_action = train.groupby(['session', 'aid'])['ts'].max().reset_index()
    train_last_ts = train.groupby(['session'])['ts'].max().reset_index()
    train_last_action = train_last_action.merge(train_last_ts, on = 'session', how = 'left')
    train_last_action['last_action_diff_hour'] = (train_last_action['ts_y'] - train_last_action['ts_x']) / 3600
    train_last_action = train_last_action[['session', 'aid', 'last_action_diff_hour']]
    
    session_aid_df = session_aid_df.merge(train_last_1hour_click, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_1hour_cart, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_1hour_order, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_1day_click, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_1day_cart, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_1day_order, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_1week_click, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_1week_cart, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_1week_order, on = ['session', 'aid'], how = 'left')
    session_aid_df = session_aid_df.merge(train_last_action, on = ['session', 'aid'], how = 'left')
    
    session_aid_df = session_aid_df.to_pandas()
    session_aid_df.fillna(0, inplace = True)
    
    session_aid_df.to_parquet(output_path + prefix + 'session_aid_df.parquet')
    del session_aid_df


def session_action_feature(prefix, train, output_path):
    
    aid_features = cudf.DataFrame(pd.read_parquet(output_path + prefix + 'aid_features_df.parquet'))
    train_dup = train[['session', 'aid', 'ts', 'type']].drop_duplicates().reset_index(drop=True)
    train_dup = train_dup.merge(aid_features[['aid', 'cart_cvr_unique', 'click_cvr_unique',
                                      'click_share_all', 'aid_rank_mean']], on = 'aid', how = 'left')
    
    train_dup = train_dup.groupby('session').agg({'click_cvr_unique':['mean'], 
                                      'cart_cvr_unique':['mean'],
                                      'click_share_all':['mean'],
                                      'aid_rank_mean':['mean'],
                                     }).reset_index()
    train_dup.columns = ['session', 'session_click_cvr_unique', 'session_cart_cvr_unique', 
                         'session_click_share_all', 'session_aid_rank_mean']
    train_dup = cudf.DataFrame(train_dup.to_pandas().fillna(0))
    
    for col in ['session_click_cvr_unique', 'session_cart_cvr_unique', 
                'session_click_share_all', 'session_aid_rank_mean']:

        train_dup[col] = train_dup[col].astype(np.float32)
        
    train_dup.to_parquet(output_path + prefix + 'session_use_aid_feat_df.parquet')
    del train_dup


# ### make features

def main(train_test_path, train_validation_path, output_path):

    for prefix in ['train_', 'test_']:

        print(prefix)

        if prefix == 'test_':
            train = cudf.DataFrame(pd.read_parquet(train_test_path + 'test.parquet'))
            train_all = cudf.DataFrame(pd.read_parquet(train_test_path + 'train.parquet'))
            merge = cudf.concat([train_all, train])
            del train_all
        else:
            train = cudf.DataFrame(pd.read_parquet(train_validation_path + 'test.parquet'))
            train_all = cudf.DataFrame(pd.read_parquet(train_validation_path + 'train.parquet'))
            merge = cudf.concat([train_all, train])
            del train_all

        session_day_feature(prefix, train, output_path)
        session_feature(prefix, train, output_path)
        aid_feature(prefix, merge, train, output_path)
        last_chunk_session_aid(prefix, train, output_path)
        user_aid_feature(prefix, train, output_path)
        session_action_feature(prefix, train, output_path)


raw_opt_path = '../../input/train_test/'
preprocess_path = '../../input/train_valid/'
output_path = '../../input/feature/'

main(raw_opt_path, preprocess_path, output_path)


