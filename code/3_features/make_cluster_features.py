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


# ### clustering co-feature

def make_cluster_co_mat(merge, train):

    # make cluster co-matrix
    result_all = []
    cluster_col = 'cluster_16dim'
    
    for i in tqdm(range(merge[cluster_col].max() + 1)):
        
        row = merge[(merge[cluster_col] == i) & (train['type'] == 0)][['session', 'ts', cluster_col]]
        if len(row) > 3000000:
            row = row.sample(3000000, random_state = 1208)
        row = row.merge(train[train['type'] == 0][['session', 'ts', cluster_col]], on = 'session', how = 'inner')
        
        result = row[[cluster_col + '_x', cluster_col + '_y']].value_counts(normalize=True).reset_index()
        result.columns = [cluster_col + '_x', cluster_col + '_y', cluster_col + '_trans_prob']
        result_all.append(result)
        
    cluster_co_matrix = cudf.concat(result_all)

    return cluster_co_matrix


def main(cluster_path, raw_opt_path, preprocess_path, candidate_path, datamart_path):
    
    cluster_16dims = cudf.read_parquet(cluster_path + 'aid_cluster.parquet')
    cluster_16dims.columns = ['aid', 'cluster_16dim']

    for prefix in ['train_', 'test_']:

        print(prefix)

        if prefix == 'test_':
            train = cudf.read_parquet(raw_opt_path + 'train.parquet')
            test = cudf.read_parquet(raw_opt_path + 'test.parquet')
            merge = cudf.concat([train, test]).reset_index(drop=True)
            del train
            gc.collect()
        else:
            train = cudf.read_parquet(preprocess_path + 'train.parquet')
            test = cudf.read_parquet(preprocess_path + 'test.parquet')
            merge = cudf.concat([train, test]).reset_index(drop=True)
            del train
            gc.collect()

        merge = merge.merge(cluster_16dims, on = 'aid', how = 'left')
        test = test.merge(cluster_16dims, on = 'aid', how = 'left')

        cluster_co_matrix = make_cluster_co_mat(merge, test)

        del merge
        gc.collect()

        # test last aid
        test_last_aid = test.groupby('session')['aid'].last().reset_index()

        for candidate_file_name in ['click_candidate.parquet', 'cart_candidate.parquet', 'order_candidate.parquet']:

            print(candidate_file_name)
            candidate = cudf.read_parquet(candidate_path + prefix + candidate_file_name)
            type_name = candidate_file_name.split('_')[0] + '_'

            if prefix == 'train_':
                del candidate['target']

            candidate['session'] = candidate['session'].astype(np.int32)
            candidate['aid'] = candidate['aid'].astype(np.int32)
            candidate = candidate.merge(test_last_aid, on = 'session', how = 'left')
            candidate = candidate.merge(cluster_16dims, left_on = 'aid_x', right_on = 'aid', how = 'left')
            candidate = candidate.merge(cluster_16dims, left_on = 'aid_y', right_on = 'aid', how = 'left')
            candidate = candidate.merge(cluster_co_matrix, on = ['cluster_16dim_x', 'cluster_16dim_y'], how = 'left')
            del candidate['cluster_16dim_x'], candidate['cluster_16dim_y'], candidate['aid_y']
            gc.collect()

            candidate['cluster_16dim_trans_prob'] = candidate['cluster_16dim_trans_prob'].astype(np.float32)
            candidate.columns = ['session', 'aid', 'cluster_16dim_trans_prob']

            candidate.to_pandas().to_parquet(datamart_path + prefix + type_name + 'cluster_trans_prob.parquet')
            del candidate
            gc.collect()


raw_opt_path = '../../input/train_test/'
preprocess_path = '../../input/train_valid/'
candidate_path = '../../input/candidate/'
datamart_path = '../../input/feature/'
cluster_path = '../../input/preprocess/'

main(cluster_path, raw_opt_path, preprocess_path, candidate_path, datamart_path)
