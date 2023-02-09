# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# - get the candidate and the probability of transitioning to the same id from the co-matrix

import xgboost as xgb
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
import itertools
from collections import Counter
import polars as pl
import cuml, cupy
from cuml.neighbors import NearestNeighbors

import cudf, itertools
print('We will use RAPIDS version',cudf.__version__)
cudf.set_option("default_integer_bitwidth", 32)
cudf.set_option("default_float_bitwidth", 32)

raw_opt_path = '../../data/raw_opt/'
preprocess_path = '../../data/val_radek/'
cd_save_path = '../../output_intermediate/candidate/'
feature_save_path = '../../output_intermediate/feature/'


# ### load data

def read_data(train_or_test, raw_opt_path, preprocess_path, get_type):

    if train_or_test == 'test':
        train = cudf.read_parquet(raw_opt_path + 'train.parquet')
        test = cudf.read_parquet(raw_opt_path + 'test.parquet')
        train = train.sort_values(['session', 'ts'])
        test = test.sort_values(['session', 'ts'])
        merge = cudf.concat([train, test])
        prefix = 'test_'
    else:
        train = cudf.read_parquet(preprocess_path + 'train.parquet')
        test = cudf.read_parquet(preprocess_path + 'test.parquet')
        train = train.sort_values(['session', 'ts'])
        test = test.sort_values(['session', 'ts'])
        merge = cudf.concat([train, test])
        prefix = 'train_'
    
    gc.collect()
        
    if get_type == 'train':
        return train
    elif get_type == 'test':
        return test
    elif get_type == 'merge':
        return merge
    else:
        return train, test, merge


def make_action(train):
    
    # last_action
    train_max_ts = train.groupby('session')['ts'].max().reset_index()
    train_last_action = train.merge(train_max_ts, on = 'session', how = 'left')
    train_last_action = train_last_action[train_last_action['ts_x'] == train_last_action['ts_y']][['session', 'aid']].drop_duplicates()
    
    # top_click
    train_top_click = train[train['type'] == 0][['session', 'aid']].value_counts().reset_index()
    train_top_click.columns = ['session', 'aid', 'n']
    train_top_click['share'] = train_top_click.groupby('session')['n'].rank(ascending=False, pct=True, method='max')
    train_top_click = train_top_click[train_top_click['share'] <= 0.3]
    train_top_click['weight'] = 1 - train_top_click['share'] 
    train_top_click = train_top_click[['session', 'aid']].drop_duplicates()
    
    # last1hour
    last_ts = train.groupby('session')['ts'].max().reset_index()
    last_ts['ts_hour'] = last_ts['ts'] - (1 * 60 * 60)
    last_ts['ts_day'] = last_ts['ts'] - (24 * 60 * 60)
    last_ts['ts_week'] = last_ts['ts'] - (7 * 24 * 60 * 60)
    last_ts.columns = ['session', 'ts_max', 'ts_hour', 'ts_day', 'ts_week']
    train_last = train.merge(last_ts, on = ['session'], how = 'left')
    train_last = train_last.drop_duplicates()
    
    # for click
    train_hour = train_last[(train_last['ts'] >= train_last['ts_hour']) & (train_last['ts'] != train_last['ts_max'])]
    train_day = train_last[(train_last['ts'] >= train_last['ts_day']) & (train_last['ts'] < train_last['ts_hour'])]
     
    # for cart
    train_buy = train[train['type'] != 0]
    
    return train_last_action, train_top_click, train_hour, train_day, train_buy


# ### make co-matrix

# #### preprocess

def make_cart_cvr(merge):

    chunk = 7000
    co_matrix = []
    cart_cvr_df = []
    chunk_num = int(len(merge['aid'].drop_duplicates()) / chunk) + 1

    for i in tqdm(range(chunk_num)):

        start = i * chunk
        end = (i + 1) * chunk

        row = merge[(merge['aid'] >= start) & (merge['aid'] < end) & (merge['type'] == 0)]
        row_cart = row.merge(merge[merge['type'] == 1], on = 'session', how = 'inner')

        click_all = row[['aid', 'session']].drop_duplicates()['aid'].value_counts().reset_index()
        click_cart = row_cart[['session', 'aid_x']].drop_duplicates()['aid_x'].value_counts().reset_index()
        click_all.columns = ['aid', 'click_n']
        click_cart.columns = ['aid', 'cart_n']
        click_all = click_all.merge(click_cart, on = 'aid', how = 'left')
        click_all['cart_cvr'] = (click_all['cart_n'] / click_all['click_n']).round(5)  

        cart_cvr_df.append(click_all)

    cart_cvr_df = cudf.concat(cart_cvr_df)

    del click_all, click_cart

    cart_cvr_df = cudf.DataFrame(cart_cvr_df.to_pandas().fillna(0))
    # 3以下はcvrにmeanをかける
    mean_cvr = cart_cvr_df['cart_cvr'].mean()
    cart_cvr_df['cart_cvr'] = np.where(cart_cvr_df['click_n'].to_pandas() < 4,  
                                       cart_cvr_df['cart_cvr'].to_pandas() * mean_cvr, 
                                       cart_cvr_df['cart_cvr'].to_pandas())
    del merge
    gc.collect()
    
    return cart_cvr_df


# ### Co-matrix

def get_use_aids(make_pattern, merge, start_type, end_type, chunk = 20000, cutline = 20):

    co_matrix = []
    aid_count_df = []
    chunk_num = int(len(merge['aid'].drop_duplicates()) / chunk) + 1
    #chunk_num = 1
    
    # check n
    for i in tqdm(range(chunk_num)):

        start = i * chunk
        end = (i + 1) * chunk
        
        if start_type == 'click':
            row = merge[(merge['aid'] >= start) & (merge['aid'] < end) & (merge['type'] == 0)]
        else:
            row = merge[(merge['aid'] >= start) & (merge['aid'] < end) & (merge['type'] != 0)]
            
        gc.collect()
            
        if end_type == 'click':
            row = row.merge(merge[merge['type'] == 0], on = 'session', how = 'inner')
        else:
            row = row.merge(merge[merge['type'] != 0], on = 'session', how = 'inner')
        
        # make pattern
        if make_pattern == 'allterm':
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
        elif make_pattern == 'base':
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
        elif make_pattern == 'base_wlen':
            row = row[row['ts_y'] - row['ts_x'] >= 0]
        elif make_pattern == 'base_hour':
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[row['ts_y'] - row['ts_x'] <= 3600]
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
        elif make_pattern == 'dup':
            pass
        elif make_pattern == 'dup_wlen':
            pass
        elif make_pattern == 'dup_hour':
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[row['ts_y'] - row['ts_x'] <= 3600]
        else:
            pass

        aid_count = row['aid_x'].value_counts().reset_index()
        aid_count.columns = ['aid', 'n']
        aid_count_df.append(aid_count)

    aid_count_df = cudf.concat(aid_count_df)

    del row, aid_count
    gc.collect()
    
    # get high association Rule aids
    low_count_aids = list(aid_count_df[aid_count_df['n'] < cutline]['aid'].to_pandas())
    high_count_aids = list(aid_count_df[aid_count_df['n'] >= cutline]['aid'].to_pandas())
    print('low:', len(low_count_aids), ' high:', len(high_count_aids))
    
    return low_count_aids, high_count_aids, aid_count_df


def aug_data(make_pattern, co_matrix_df, cart_cvr_df, base_co_matrix_df):

    co_matrix_df['share'] = co_matrix_df['share'].astype(np.float32)
    co_matrix_df['cart_cvr'] = co_matrix_df['cart_cvr'].astype(np.float32)
    
    # get high asso aids
    high_asso_df = base_co_matrix_df[base_co_matrix_df['rank'] <= 2]
    print('high asso len:', len(high_asso_df))
    high_rank_df = co_matrix_df[co_matrix_df['rank'] <= 3]
    del co_matrix_df['rank']
    gc.collect()
    
    print('high_rank_df len:', len(high_rank_df))
    aug_data = high_rank_df.merge(high_asso_df[['aid_x', 'aid_y', 'share']],
                              left_on = ['aid_y'], 
                              right_on = ['aid_x'], how = 'inner').sort_values(['aid_x_x'])
    aug_data = aug_data[aug_data['aid_x_x'] != aug_data['aid_y_y']]
    aug_data = aug_data[['aid_x_x', 'aid_y_y', 'share_x', 'share_y']]
    aug_data['share'] = (aug_data['share_x'] * aug_data['share_y']).round(5)
    aug_data = aug_data[['aid_x_x', 'aid_y_y', 'share']]
    aug_data.columns = ['aid_x', 'aid_y', 'share']
    aug_data = aug_data.groupby(['aid_x', 'aid_y'])['share'].sum().reset_index()

    print('before:', co_matrix_df.shape[0], aug_data.shape[0], co_matrix_df.shape[0] + aug_data.shape[0])
    # merge
    co_matrix_df = cudf.concat([co_matrix_df[['aid_x', 'aid_y', 'share']], aug_data[['aid_x', 'aid_y', 'share']]])
    del aug_data
    gc.collect()
    
    co_matrix_df = co_matrix_df.groupby(['aid_x', 'aid_y'])['share'].max().reset_index()
    co_matrix_df = co_matrix_df.merge(cart_cvr_df[['aid', 'cart_cvr']], left_on = ['aid_y'], 
                                      right_on = ['aid'], how = 'left')
    #print('check dup:', co_matrix_df[['aid_x', 'aid_y']].value_counts().value_counts())
    co_matrix_df = co_matrix_df.reset_index(drop=True)
    print('after:', co_matrix_df.shape)
    
    return co_matrix_df


def make_chunk_co_matrix(make_pattern, merge, use_aids, cart_cvr_df, start_type, end_type, chunk, same_col_share_name, cut_rank):

    co_matrix_df = []
    co_matrix_same_df = []
    num = 0
    chunk_num = int(len(use_aids) / chunk) + 1
    #chunk_num = 1
    
    for i in tqdm(range(chunk_num)):
        start = i * chunk
        end = (i + 1) * chunk
        
        # extract data
        if start_type == 'click':
            row = merge[(merge['aid'].isin(use_aids[start:end])) & (merge['type'] == 0)]
        else:
            row = merge[(merge['aid'].isin(use_aids[start:end])) & (merge['type'] != 0)]
            
        gc.collect()
            
        if end_type == 'click':
            row = row.merge(merge[merge['type'] == 0], on = 'session', how = 'inner')
        else:
            row = row.merge(merge[merge['type'] != 0], on = 'session', how = 'inner')
            
        # get many pattern co-matrix 
        if make_pattern == 'allterm':
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
            
        elif make_pattern == 'base':   
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
        
        elif make_pattern == 'base_wlen':
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row['ts_diff'] = np.abs(row['ts_y'] - row['ts_x'])
            row['diff_rank'] = row.groupby(['session', 'aid_x'])['ts_diff'].rank(method = 'min')
            row['diff_weight'] = 1 / row['diff_rank']
            del row['ts_diff'], row['diff_rank']
            row = row.groupby(['aid_x', 'aid_y'])['diff_weight'].sum().reset_index()
            
        elif make_pattern == 'base_hour':
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[row['ts_y'] - row['ts_x'] <= 3600]
            row = row[['session', 'aid_x', 'aid_y']].drop_duplicates()
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
            
        elif make_pattern == 'dup':
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
            
        elif make_pattern == 'dup_wlen':
            row['ts_diff'] = np.abs(row['ts_y'] - row['ts_x'])
            row['diff_rank'] = row.groupby(['session', 'aid_x'])['ts_diff'].rank(method = 'min')
            row['diff_weight'] = 1 / row['diff_rank']
            del row['ts_diff'], row['diff_rank']
            row = row.groupby(['aid_x', 'aid_y'])['diff_weight'].sum().reset_index()
        
        elif make_pattern == 'dup_hour':
            row = row[row['ts_y'] - row['ts_x'] >= 0]
            row = row[row['ts_y'] - row['ts_x'] <= 3600]
            row = row[['aid_x', 'aid_y']].value_counts().reset_index().sort_values(['aid_x'])
            
        row.columns = ['aid_x','aid_y','n']
        aid_x_total = row.groupby(['aid_x'])['n'].sum().reset_index()
        aid_x_total.columns = ['aid_x', 'aid_total']
        row = row.merge(aid_x_total, on = 'aid_x', how = 'left')
        
        del aid_x_total
        gc.collect()
        
        row['share'] = row['n'] / row['aid_total']
        row['share'] = row['share'].astype(np.float32)

        # cart_cvrと紐づけ
        row = row[['aid_x', 'aid_y', 'share']].merge(cart_cvr_df[['aid', 'cart_cvr']], 
                                                     left_on = ['aid_y'], right_on = ['aid'], how = 'left')
        row = row[['aid_x', 'aid_y', 'share', 'cart_cvr']].sort_values(
            ['aid_x', 'share', 'cart_cvr', 'aid_y'], ascending = [True, False, False, True]).reset_index(drop=True)

        # ranking
        row['rank'] = 1
        row['rank'] = row.groupby('aid_x')['rank'].cumsum()
        row['rank'] = row['rank'].astype(np.int32)
        row_same_aid = row[row['aid_x'] == row['aid_y']] 
        row = row[row['aid_x'] != row['aid_y']] 
        row = row[row['rank'] <= cut_rank]
        
        #print(row.head(5))
        #print(row.dtypes)

        num += len(row)
        print(num)
        co_matrix_df.append(row)
        co_matrix_same_df.append(row_same_aid)
    
    co_matrix_df = cudf.concat(co_matrix_df)
    co_matrix_same_df = cudf.concat(co_matrix_same_df)
    co_matrix_same_df = co_matrix_same_df[['aid_x', 'share']]
    co_matrix_same_df.columns = ['aid', same_col_share_name]
    co_matrix_same_df[same_col_share_name] = co_matrix_same_df[same_col_share_name].astype(np.float32)
    
    return co_matrix_df, co_matrix_same_df


def make_co_matrix(train_or_test, make_pattern, cart_cvr_df, start_type, end_type, same_col_share_name, 
                   chunk = 7000, cutline = 20, cut_rank = 151):
    
    merge = read_data(train_or_test, raw_opt_path, preprocess_path, get_type = 'merge')
    
    print('check data size...')
    low_count_aids, high_count_aids, aid_count_df = get_use_aids(make_pattern, merge, start_type, end_type, chunk, cutline)
    
    print('make high prob aids...')
    high_co_matrix, high_same_aids = make_chunk_co_matrix(make_pattern, merge, high_count_aids, cart_cvr_df, 
                                                          start_type, end_type, chunk, same_col_share_name, cut_rank)
    base_aug = high_co_matrix.copy()
    del merge # memory management
    gc.collect()
    
    print('high co-matrix shape:', high_co_matrix.shape)
    print('aug...')
    high_co_matrix = aug_data(make_pattern, high_co_matrix, cart_cvr_df, base_aug)
    
    print('make low prob aids...')
    merge = read_data(train_or_test, raw_opt_path, preprocess_path, get_type = 'merge')
    low_co_matrix, low_same_aids = make_chunk_co_matrix(make_pattern, merge, low_count_aids, cart_cvr_df, 
                                                        start_type, end_type, chunk, same_col_share_name, cut_rank)
    del merge # memory management
    gc.collect()
    
    low_co_matrix = aug_data(make_pattern, low_co_matrix, cart_cvr_df, base_aug)
    print('after data size:', low_co_matrix.shape)
    
    del base_aug
    gc.collect()
    
    high_co_matrix = cudf.concat([high_co_matrix, low_co_matrix])
    high_co_matrix['share'] = high_co_matrix['share'].astype(np.float32)
    high_co_matrix['cart_cvr'] = high_co_matrix['cart_cvr'].astype(np.float32)
    del low_co_matrix, high_co_matrix['aid']
    gc.collect()
    
    high_same_aids  = cudf.concat([high_same_aids, low_same_aids])
    print('total size:', high_co_matrix.shape)
    
    return high_co_matrix, high_same_aids


def make_action_datamart(co_matrix, session_action_df, co_mat_feature_name, rank, dm_save_path, prefix, w2v = False):
    
    action_df = session_action_df.copy()
    del session_action_df
    gc.collect()
    
    if w2v == False:
        action_df = action_df.merge(co_matrix, left_on = 'aid', right_on = 'aid_x', how = 'inner')
        action_df = action_df.groupby(['session', 'aid_y']).agg({'share':'sum', 'cart_cvr': 'mean'}).reset_index()
        action_df = action_df.sort_values(['session', 'share', 'cart_cvr', 'aid_y'], 
                                          ascending = [True, False, False, True])
    else:
        action_df = action_df.merge(co_matrix, left_on = 'aid', right_on = 'aid_x', how = 'inner')
        action_df = action_df.groupby(['session', 'aid_y'])['share'].mean().reset_index()
        action_df = action_df.sort_values(['session', 'share']).reset_index(drop=True)
        
    action_df = action_df[['session', 'aid_y', 'share']]
    action_df['rank'] = 1
    action_df['rank'] = action_df.groupby('session')['rank'].cumsum()
    action_df['rank'] = action_df['rank'].astype(np.int32)
    action_df = action_df[action_df['rank'] <= rank]
    action_df.columns = ['session', 'aid',  co_mat_feature_name, 'rank']
    action_df.to_pandas().to_parquet(dm_save_path + prefix + co_mat_feature_name + '.parquet')
    
    del action_df
    gc.collect()


def make_action_hour_day_datamart(co_matrix, train_action, co_mat_feature_name, cut_datamart, dm_save_path, prefix, w2v = False):
    
    chunk = 20000
    chunk_num = int(len(train_action['session'].drop_duplicates()) / chunk) + 1
    chunk_num

    datamart_list = []
    row_len = 0
    session_list = list(train_action['session'].unique().to_pandas())

    for i in tqdm(range(chunk_num)):

        start = i * chunk
        end = (i + 1) * chunk

        row = train_action[train_action['session'].isin(session_list[start:end])].merge(co_matrix, left_on = 'aid', right_on = 'aid_x')
        
        if w2v == False:
            row = row.groupby(['session', 'aid_y'])['share'].sum().reset_index()
            row = row.sort_values(['session', 'share'], ascending = [True, False])
        else:
            row = row.groupby(['session', 'aid_y'])['share'].mean().reset_index()
            row = row.sort_values(['session', 'share']).reset_index(drop=True)
            
        row['rank'] = 1
        row['rank'] = row.groupby('session')['rank'].cumsum()
        row = row[row['rank'] <= cut_datamart]
    
        row_len += len(row)
        print(row_len)

        datamart_list.append(row)

    datamart = cudf.concat(datamart_list)
    datamart.columns = ['session', 'aid', co_mat_feature_name, 'rank']
    datamart.to_pandas().to_parquet(dm_save_path + prefix + co_mat_feature_name + '.parquet')
    del datamart
    gc.collect()


def w2v_co_matrix(w2v_path):

    w2v = cudf.DataFrame(pd.read_parquet(w2v_path + 'test_' + 'w2v_output_16dims.parquet'))
    w2v = w2v.sort_values('aid').reset_index(drop=True)

    KNN = 60
    model_knn = NearestNeighbors(n_neighbors=KNN, metric = 'cosine')
    model_knn.fit(w2v.iloc[:, 1:])
    distances, indices = model_knn.kneighbors(w2v.iloc[:, 1:])

    co_matrix = cudf.DataFrame(np.array(([[i] * KNN for i in tqdm(range(0,len(w2v)))])).reshape(-1), columns = ['aid_x'])
    co_matrix['aid_y'] = np.array(indices.to_pandas()).reshape(-1)
    co_matrix['dist'] = np.array(distances.to_pandas()).reshape(-1)
    co_matrix = co_matrix[co_matrix['aid_x'] != co_matrix['aid_y']]

    co_matrix.columns = ['aid_x', 'aid_y', 'share']

    co_matrix['aid_x'] = co_matrix['aid_x'].astype(np.int32)
    co_matrix['aid_y'] = co_matrix['aid_y'].astype(np.int32)
    co_matrix = co_matrix.sort_values(['aid_x', 'share', 'aid_y'], ascending = [True, True, True])

    del distances, indices
    gc.collect()
    
    return co_matrix


def main(raw_opt_path, preprocess_path, dm_save_path, w2v_path):
    
    for train_or_test in ['test', 'train']:
        
        print('start:', train_or_test)
    
        if train_or_test == 'train':
            prefix = 'train_'
        else:
            prefix = 'test_'

        train = read_data(train_or_test, raw_opt_path, preprocess_path, get_type = 'test')
        train_last_action, train_top_action, train_hour, train_day, train_buy = make_action(train)
        merge = read_data(train_or_test, raw_opt_path, preprocess_path, get_type = 'merge')
        same_aid_df = merge[['aid']].drop_duplicates().reset_index(drop=True).to_pandas()
        cart_cvr_df = make_cart_cvr(merge)
        del merge
        gc.collect()
        
        co_dict = {'allterm': [['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']], 
                       ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']] , 
                       ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
                       ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]],
            'dup': [['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']], 
                       ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']] , 
                       ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
                       ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]],
            'dup_wlen': [['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']], 
                       ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']] , 
                       ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
                       ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]],
            'dup_hour': [['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'hour']], 
                       ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'hour']] , 
                       ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
                       ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]],
            'base': [['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']], 
                       ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']] , 
                       ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
                       ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]],
            'base_wlen': [['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']], 
                       ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']] , 
                       ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
                       ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]],
            'base_hour': [['click', 'click', 20, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']], 
                       ['click', 'buy', 10, 101, 100, 200, 250, 200, ['last', 'top', 'hour', 'day']] , 
                       ['buy', 'click', 10, 101, 100, 200, 250, 200, ['all']],
                       ['buy', 'buy', 10, 101, 100, 200, 250, 200, ['all']]],
            'w2v': [['click', 'click', 20, 60, 50, 50, 50, 50, ['last', 'hour']]]
                  }
                
        print('start co-mat datamart')
        for i in co_dict.keys():
            print(i)
            make_pattern = i

            for j in range(len(co_dict[i])):

                print(co_dict[i][j])
                start_type = co_dict[i][j][0]
                end_type = co_dict[i][j][1]
                cutline = co_dict[i][j][2]
                cut_rank = co_dict[i][j][3]
                cut_datamart_last = co_dict[i][j][4]
                cut_datamart_top = co_dict[i][j][5]
                cut_datamart_hour = co_dict[i][j][6]
                cut_datamart_day = co_dict[i][j][7]
                action_pattern_list = co_dict[i][j][8]

                if make_pattern == 'w2v':
                    print('make w2v co_mat')
                    co_matrix = w2v_co_matrix(w2v_path)

                    if 'last'in action_pattern_list:
                        co_mat_feature_name = f'{start_type}_{end_type}_{make_pattern}_last_w2v'
                        make_action_datamart(co_matrix, train_last_action, co_mat_feature_name, 
                                             cut_datamart_last, dm_save_path, prefix, w2v = True)

                    if 'hour'in action_pattern_list:
                        co_mat_feature_name = f'{start_type}_{end_type}_{make_pattern}_hour_w2v'
                        make_action_hour_day_datamart(co_matrix, train_hour, co_mat_feature_name, 
                                                      cut_datamart_hour, dm_save_path, prefix, w2v = True)

                else:
                    print('make co_mat')
                    same_feature_name = f'same_{start_type}_{end_type}_{make_pattern}'        
                    co_matrix, same_feature = make_co_matrix(train_or_test, make_pattern, cart_cvr_df, 
                                                             start_type, end_type, same_feature_name, 
                                                             chunk = 20000, cutline = cutline, cut_rank = cut_rank)
                    gc.collect()
                    same_feature = same_feature.to_pandas()
                    same_aid_df = same_aid_df.merge(same_feature, on = 'aid', how = 'left')

                    if 'last'in action_pattern_list:
                        co_mat_feature_name = f'{start_type}_{end_type}_{make_pattern}_last'
                        make_action_datamart(co_matrix, train_last_action, co_mat_feature_name, 
                                             cut_datamart_last, dm_save_path, prefix)

                    if 'top'in action_pattern_list:
                        co_mat_feature_name = f'{start_type}_{end_type}_{make_pattern}_top'
                        make_action_datamart(co_matrix, train_top_action, co_mat_feature_name, 
                                             cut_datamart_top, dm_save_path, prefix)

                    if 'hour'in action_pattern_list:
                        co_mat_feature_name = f'{start_type}_{end_type}_{make_pattern}_hour'
                        make_action_hour_day_datamart(co_matrix, train_hour, co_mat_feature_name, 
                                                      cut_datamart_hour, dm_save_path, prefix)

                    if 'day'in action_pattern_list:
                        co_mat_feature_name = f'{start_type}_{end_type}_{make_pattern}_day'
                        make_action_hour_day_datamart(co_matrix, train_day, co_mat_feature_name, 
                                                      cut_datamart_day, dm_save_path, prefix)

                    if 'all'in action_pattern_list:
                        co_mat_feature_name = f'{start_type}_{end_type}_{make_pattern}_all'
                        make_action_datamart(co_matrix, train_buy, co_mat_feature_name, 
                                                      200, dm_save_path, prefix)

                    gc.collect()
                    
        same_aid_df.to_parquet(dm_save_path + prefix + 'same_aid_df.parquet')


# set path
raw_opt_path = '../../input/train_test/'
preprocess_path = '../../input/train_valid/'
dm_save_path = '../../input/feature/'
w2v_path = '../../input/preprocess/'

main(raw_opt_path, preprocess_path, dm_save_path, w2v_path)




