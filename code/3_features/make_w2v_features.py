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


def make_action_data(train):
    # last_action
    train_action_max_ts = train.groupby('session')['ts'].max().reset_index()
    train_last_action = train.merge(train_action_max_ts, on = 'session', how = 'left')
    train_last_action = train_last_action[train_last_action['ts_x'] == train_last_action['ts_y']][
        ['session', 'aid']].drop_duplicates().sort_values(['session', 'aid']).reset_index(drop=True)
    train_last_action.columns = ['session', 'target_aid']
    
    # last 1hour
    last_ts = train.groupby('session')['ts'].max().reset_index()
    last_ts['ts_hour'] = last_ts['ts'] - (1 * 60 * 60)
    last_ts['ts_day'] = last_ts['ts'] - (24 * 60 * 60)
    last_ts['ts_week'] = last_ts['ts'] - (7 * 24 * 60 * 60)
    last_ts.columns = ['session', 'ts_max', 'ts_hour', 'ts_day', 'ts_week']
    train_last = train.merge(last_ts, on = ['session'], how = 'left')
    train_hour = train_last[(train_last['ts'] >= train_last['ts_hour']) & (train_last['ts'] != train_last['ts_max'])]
    train_hour = train_hour[['session', 'aid']].drop_duplicates().sort_values(['session', 'aid']).reset_index(drop=True)
    train_hour.columns = ['session', 'target_aid']

    del train_last
    
    return train_last_action, train_hour


# +
def cos_similarity(X, Y):
    cos_sim = (X * Y).sum(axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))
    return cos_sim.reshape(-1, 1)

def w2v_session_dist_feature(prefix, train, w2v_path, candidate_path, output_path, candidate_file_name, w2v_file_name, feature_name, save_name, chunk_size = 20000):
    
    w2v = cudf.read_parquet(w2v_path + 'test_' + w2v_file_name)
    w2v = w2v.sort_values('aid').reset_index(drop=True)
    
    # train merge w2v
    session_w2v = train[['aid', 'session']]
    session_w2v = session_w2v.merge(w2v, on = 'aid', how = 'left')
    session_w2v = session_w2v.iloc[:,1:]
    session_w2v = session_w2v.groupby('session').mean().reset_index()
    
    # get chunk num
    candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
    session_list = list(candidate['session'].unique().to_pandas())
    chunk_num = int(len(session_list) / chunk_size) + 1

    del candidate
    gc.collect()
    
    # make feature
    cos_sim_list = []

    for i in tqdm(range(chunk_num)):

        start = i * chunk_size
        end = (i + 1) * chunk_size

        chunk_candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
        chunk_candidate['session'] = chunk_candidate['session'].astype(np.int32)
        chunk_candidate['aid'] = chunk_candidate['aid'].astype(np.int32)
        chunk_candidate = chunk_candidate[chunk_candidate['session'].isin(session_list[start:end])][['session', 'aid']]

        aid_vec = chunk_candidate[['session', 'aid']].merge(w2v, 
                                                            on = 'aid', 
                                                            how = 'left').sort_values(['session', 'aid']).iloc[:,2:]

        target_aid_vec = chunk_candidate[['session']].merge(session_w2v, on = 'session', 
                                                            how = 'left').sort_values(['session']).iloc[:,1:]
        chunk_candidate[feature_name] = cos_similarity(aid_vec.values, target_aid_vec.values)
        chunk_candidate = chunk_candidate.to_pandas()
        cos_sim_list.append(chunk_candidate)

        del chunk_candidate
        gc.collect()

    candidate = pd.concat(cos_sim_list)
    candidate[feature_name] = candidate[feature_name].astype(np.float32)
    candidate.to_parquet(output_path + save_name + '.parquet')
    gc.collect()


# -

def w2v_aid_dist_feature(prefix, train_action, w2v_path, candidate_path, output_path, candidate_file_name, w2v_file_name, feature_name, save_name, chunk_size = 30000):

    # read candidate and w2v
    w2v = cudf.read_parquet(w2v_path + 'test_' + w2v_file_name)
    w2v = w2v.sort_values('aid').reset_index(drop=True)
    
    candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
    session_list = list(candidate['session'].unique().to_pandas())
    chunk_num = int(len(session_list) / chunk_size) + 1
    del candidate
    gc.collect()

    cos_sim_list = []

    for i in tqdm(range(chunk_num)):

        start = i * chunk_size
        end = (i + 1) * chunk_size
        
        chunk_candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
        chunk_candidate['session'] = chunk_candidate['session'].astype(np.int32)
        chunk_candidate['aid'] = chunk_candidate['aid'].astype(np.int32)
        chunk_candidate = chunk_candidate[chunk_candidate['session'].isin(session_list[start:end])][['session', 'aid']]
        gc.collect()
        
        chunk_candidate = chunk_candidate.merge(train_action, on = 'session', how = 'inner')
        chunk_candidate = chunk_candidate.sort_values(['session', 'aid']).reset_index(drop=True)
        aid_vec = chunk_candidate[['session', 'aid']].merge(w2v, on = 'aid', how = 'left').sort_values(
            ['session', 'aid']).iloc[:,2:]
        target_aid_vec = chunk_candidate[['session', 'target_aid']].merge(
            w2v, left_on = 'target_aid', right_on = 'aid', how = 'left').sort_values(['session', 'aid']).iloc[:,3:]
        
        chunk_candidate[feature_name] = cos_similarity(aid_vec.values, target_aid_vec.values)
        chunk_candidate = chunk_candidate.groupby(['session', 'aid'])[feature_name].mean().reset_index()
        chunk_candidate = chunk_candidate.to_pandas()
        cos_sim_list.append(chunk_candidate)
        
        del chunk_candidate
        gc.collect()

    candidate = pd.concat(cos_sim_list)
    candidate[feature_name] = candidate[feature_name].astype(np.float32)
    candidate.to_parquet(output_path + save_name + '.parquet')
    gc.collect()


def main(raw_opt_path, preprocess_path, w2v_path, candidate_path, output_path, candidate_file_name, prefix):

    #for prefix in ['train_', 'test_']:

    print(prefix)

    if prefix == 'test_':
        train = cudf.read_parquet(raw_opt_path + 'test.parquet')
    else:
        train = cudf.read_parquet(preprocess_path + 'test.parquet')

    print(candidate_file_name)
    type_name = candidate_file_name.split('_')[0] + '_'

    w2v_session_dist_feature(prefix, train, w2v_path, candidate_path, output_path, candidate_file_name, 
                             w2v_file_name = 'w2v_output_16dims.parquet', 
                             feature_name = 'session_w2v_dist_16dim', 
                             save_name = prefix + type_name + 'session_w2v_dist_16dim',
                             chunk_size = 15000)
    w2v_session_dist_feature(prefix, train, w2v_path, candidate_path, output_path, candidate_file_name, 
                             w2v_file_name = 'w2v_output_64dims.parquet', 
                             feature_name = 'session_w2v_dist_64dim', 
                             save_name = prefix + type_name + 'session_w2v_dist_64dim',
                             chunk_size = 15000)
    gc.collect()

    train_last_action, train_hour = make_action_data(train)

    w2v_aid_dist_feature(prefix, train_last_action, w2v_path, candidate_path, output_path, candidate_file_name, 
                         w2v_file_name = 'w2v_output_16dims.parquet',
                         feature_name = 'aid_w2v_last_dist_16dim', 
                         save_name = prefix + type_name + 'aid_w2v_last_dist_16dim',
                         chunk_size = 15000)

    w2v_aid_dist_feature(prefix, train_last_action, w2v_path, candidate_path, output_path, candidate_file_name, 
                         w2v_file_name = 'w2v_output_64dims.parquet',
                         feature_name = 'aid_w2v_last_dist_64dim', 
                         save_name = prefix + type_name + 'aid_w2v_last_dist_64dim',
                         chunk_size = 15000)

    w2v_aid_dist_feature(prefix, train_hour, w2v_path, candidate_path, output_path, candidate_file_name, 
                         w2v_file_name = 'w2v_output_16dims.parquet',
                         feature_name = 'aid_w2v_hour_dist_16dim', 
                         save_name = prefix + type_name + 'aid_w2v_hour_dist_16dim',
                         chunk_size = 10000)

    w2v_aid_dist_feature(prefix, train_hour, w2v_path, candidate_path, output_path, candidate_file_name, 
                         w2v_file_name = 'w2v_output_64dims.parquet',
                         feature_name = 'aid_w2v_hour_dist_64dim', 
                         save_name = prefix + type_name + 'aid_w2v_hour_dist_64dim',
                         chunk_size = 10000)
    gc.collect()


raw_opt_path = '../../input/train_test/'
preprocess_path = '../../input/train_valid/'
w2v_path = '../../input/preprocess/'
candidate_path = '../../input/candidate/'
output_path = '../../input/feature/'

# +
main(raw_opt_path, preprocess_path, w2v_path, candidate_path, output_path, 'click_candidate.parquet', 'train_')
main(raw_opt_path, preprocess_path, w2v_path, candidate_path, output_path, 'cart_candidate.parquet', 'train_')
main(raw_opt_path, preprocess_path, w2v_path, candidate_path, output_path, 'order_candidate.parquet', 'train_')

main(raw_opt_path, preprocess_path, w2v_path, candidate_path, output_path, 'click_candidate.parquet', 'test_')
main(raw_opt_path, preprocess_path, w2v_path, candidate_path, output_path, 'cart_candidate.parquet', 'test_')
main(raw_opt_path, preprocess_path, w2v_path, candidate_path, output_path, 'order_candidate.parquet', 'test_')
# -


