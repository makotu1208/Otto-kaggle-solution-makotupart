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

# +
from catboost import CatBoostRanker, Pool, MetricVisualizer
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
import itertools
from collections import Counter
import random
import yaml

from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from xgboost import plot_importance
import polars as pl
import cudf, itertools
print('We will use RAPIDS version',cudf.__version__)
cudf.set_option("default_integer_bitwidth", 32)
cudf.set_option("default_float_bitwidth", 32)


# -

def main(type_name, model_path, candidate_path, datamart_path, oof_read_path, output_path,
         feature_dict_path, co_matrix_dict_path, oof_dict_path):
    
    prefix = 'test_'
    
    with open(feature_dict_path) as yml:
        feature_config = yaml.safe_load(yml)
        
    with open(co_matrix_dict_path) as yml:
        co_matrix_config = yaml.safe_load(yml)
    
    with open(oof_dict_path) as yml:
        oof_config = yaml.safe_load(yml)
        
    FEATURES = feature_config[f'test_{type_name}']
    co_matrix_list = co_matrix_config[f'test_{type_name}']
    oof_dict = oof_config[f'test_{type_name}']
    model_path = [model_path + '/' + i for i in os.listdir(model_path) if 'cbm' in i]
    print(model_path)
    
    train_all = pd.read_parquet(candidate_path + 'test_' + type_name + '_candidate.parquet')
    train_all.fillna(0, inplace = True)
    
    test_session_list = list(train_all['session'].unique())
    print(len(test_session_list))
    chunk_size = 335000 # 280000
    chunk_num = int((len(test_session_list) / chunk_size) + 1)
    
    if type_name == 'click_all':
        feature_type_name = 'click'
    else:
        feature_type_name = type_name
    
    print(chunk_num)
    
    pred_all = []

    for i in tqdm(range(chunk_num)):

        print('chank', i)

        start = i * chunk_size
        end = (i + 1) * chunk_size

        print('load train all...')
        train_all = pl.read_parquet(candidate_path + 'test_' + type_name + '_candidate.parquet')
        test_chunk_session = test_session_list[start:end]
        test_chunk = train_all.filter(pl.col("session").is_in(test_chunk_session))
        test_chunk = test_chunk.select(pl.all().sort_by(['session', 'aid']))
        test_chunk_aids = list(test_chunk['aid'].unique())
        test_chunk = test_chunk.with_column(pl.col(['session','aid']).cast(pl.Int32))

        del train_all
        gc.collect()
        
        print('oof features...')
        for oof_file_name in oof_dict.keys():
            print(oof_file_name)
            oof = pl.read_parquet(oof_read_path + oof_file_name)
            oof = oof.drop(['__index_level_0__'])
            oof = oof.rename({'pred': oof_dict[oof_file_name]})
            test_chunk = test_chunk.join(oof, on=['aid', 'session'], how="left")
        
        print('BPR features...')
        bpr_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '__bpr_feature.parquet')
        if 'target' in list(bpr_df.columns):
            print('drop')
            bpr_df = bpr_df.drop(['target'])
        bpr_df = bpr_df.with_column(pl.col(pl.Int64).cast(pl.Int32, strict=False))
        test_chunk = test_chunk.join(bpr_df, on=['aid', 'session'], how="left")

        print('cos-sim features...')
        cos_sim_list = ['aid_w2v_last_dist_64dim.parquet', 'aid_w2v_last_dist_16dim.parquet',
                        'aid_w2v_hour_dist_64dim.parquet', 'aid_w2v_hour_dist_16dim.parquet',
                        'session_w2v_dist_64dim.parquet', 'session_w2v_dist_16dim.parquet']

        for mat_name in cos_sim_list:
            print(mat_name)
            cos_sim_last_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_' + mat_name)
            if '__index_level_0__' in list(cos_sim_last_df.columns):
                print('drop')
                cos_sim_last_df = cos_sim_last_df.drop(['__index_level_0__'])
            #print(cos_sim_last_df)
            cos_sim_last_df = cos_sim_last_df.filter(pl.col("session").is_in(test_chunk_session))
            cos_sim_last_df = cos_sim_last_df.filter(pl.col("aid").is_in(test_chunk_aids))
            test_chunk = test_chunk.join(cos_sim_last_df, on=['aid', 'session'], how="left")
            del cos_sim_last_df
            gc.collect()
        
        print('co-mat features...')
        for co_matrix_name in co_matrix_list:
            print(co_matrix_name)
            co_matrix = pl.read_parquet(datamart_path + co_matrix_name)
            if '__index_level_0__' in list(co_matrix.columns):
                co_matrix = co_matrix.drop('__index_level_0__')
            if 'rank' in list(co_matrix.columns):
                co_matrix = co_matrix.drop('rank') #??
            co_matrix = co_matrix.filter(pl.col("session").is_in(test_chunk_session))
            co_matrix = co_matrix.filter(pl.col("aid").is_in(test_chunk_aids))
            test_chunk = test_chunk.join(co_matrix, on=['aid', 'session'], how="left")
            #df = change_dtypes(df)
            del co_matrix
            gc.collect()
    
        print('same-vec features...')
        print(mat_name)
        aid_cvr_features_df = pl.read_parquet(datamart_path + prefix + 'same_aid_df.parquet')
        aid_cvr_features_df = aid_cvr_features_df.filter(pl.col("aid").is_in(test_chunk_aids))
        # check cols & extract
        use_cols = [i for i in list(aid_cvr_features_df.columns) if i in FEATURES]
        use_cols += ['aid']
        aid_cvr_features_df = aid_cvr_features_df[use_cols]
        test_chunk = test_chunk.join(aid_cvr_features_df, on=['aid'], how="left")
        del aid_cvr_features_df 
        gc.collect()
        
        print('cluster features...')
        # cluster features
        cluster_prob_df = pl.read_parquet(datamart_path + prefix + feature_type_name + '_cluster_trans_prob.parquet')
        cluster_prob_df = cluster_prob_df.filter(pl.col("session").is_in(test_chunk_session))

        use_cols = [i for i in list(cluster_prob_df.columns) if i in FEATURES]
        use_cols += ['session', 'aid']
        test_chunk = test_chunk.join(cluster_prob_df, on=['aid', 'session'], how="left")
        del cluster_prob_df
        gc.collect()
        
        
        print('session_aid features...')
        # session / aid
        session_aid_df = pl.read_parquet(datamart_path + prefix + 'session_aid_df.parquet')
        session_aid_df = session_aid_df.filter(pl.col("session").is_in(test_chunk_session))

        # check cols & extract
        use_cols = [i for i in list(session_aid_df.columns) if i in FEATURES]
        use_cols += ['session', 'aid']
        session_aid_df = session_aid_df[use_cols]

        test_chunk = test_chunk.join(session_aid_df, on=['session', 'aid'], how="left")
        test_chunk = test_chunk.with_column(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast


        print('last chunk features...')
        # session / aid last chunk
        last_chunk_df = pl.read_parquet(datamart_path + prefix + 'last_chunk_session_aid_df.parquet')
        last_chunk_df = last_chunk_df.filter(pl.col("session").is_in(test_chunk_session))

        # check cols & extract
        use_cols = [i for i in list(last_chunk_df.columns) if i in FEATURES]
        use_cols += ['session', 'aid']
        last_chunk_df = last_chunk_df[use_cols]

        test_chunk = test_chunk.join(last_chunk_df, on=['session', 'aid'], how="left")
        test_chunk = test_chunk.with_column(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
        del last_chunk_df
        gc.collect()


        print('session features...')
        # session
        session_df = pl.read_parquet(datamart_path + prefix + 'session_df.parquet')
        session_df = session_df.filter(pl.col("session").is_in(test_chunk_session))

        # check cols & extract
        use_cols = [i for i in list(session_df.columns) if i in FEATURES]
        use_cols += ['session', 'day']
        session_df = session_df[use_cols]

        test_chunk = test_chunk.join(session_df, on=['session'], how="left")
        test_chunk = test_chunk.with_column(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
        del session_df
        gc.collect()

        print('session & use aid features...')
        # load
        session_use_aid_df = pl.read_parquet(datamart_path + prefix + 'session_use_aid_feat_df.parquet')
        session_use_aid_df = session_use_aid_df.filter(pl.col("session").is_in(test_chunk_session))

        # check cols & extract
        use_cols = [i for i in list(session_use_aid_df.columns) if i in FEATURES]
        use_cols += ['session']
        session_use_aid_df = session_use_aid_df[use_cols]

        test_chunk = test_chunk.join(session_use_aid_df, on=['session'], how="left")
        test_chunk = test_chunk.with_column(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
        del session_use_aid_df
        gc.collect()
    
        print('aid & day features...')
        # load
        session_day_df = pl.read_parquet(datamart_path + prefix + 'session_day_df.parquet')

        # check cols & extract
        use_cols = [i for i in list(session_day_df.columns) if i in FEATURES]
        use_cols += ['day', 'aid']
        session_day_df = session_day_df[use_cols]

        test_chunk = test_chunk.join(session_day_df, on= ['day', 'aid'], how="left")
        test_chunk = test_chunk.with_column(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
        del session_day_df
        gc.collect()
        test_chunk = test_chunk.drop('day')
    
        print('aid features...')
        # aid
        aid_features_df = pl.read_parquet(datamart_path + prefix + 'aid_features_df.parquet')
        aid_features_df = aid_features_df.filter(pl.col("aid").is_in(test_chunk_aids))

        # check cols & extract
        use_cols = [i for i in list(aid_features_df.columns) if i in FEATURES]
        use_cols += ['aid']
        aid_features_df = aid_features_df[use_cols]
        aid_features_df = aid_features_df.with_column(pl.col(pl.Float64).cast(pl.Float32, strict=False))
        aid_features_df = aid_features_df.with_column(pl.col(pl.Int64).cast(pl.Int32, strict=False))

        test_chunk = test_chunk.join(aid_features_df, on= ['aid'], how="left")
        test_chunk = test_chunk.with_column(pl.col(pl.Float64).cast(pl.Float32, strict=False)) # cast
        del aid_features_df
        gc.collect()

        test_chunk = test_chunk.with_column(pl.exclude('last_action_diff_hour').fill_null(0))
        test_chunk = test_chunk.to_pandas()
    
        #-------------------
        # pred
        # ------------------

        print('chunk data size:', test_chunk.shape)
        pred = test_chunk[['session', 'aid']].copy()
        result = np.zeros(len(test_chunk))

        for j in model_path:

            print(j)

            ranker = CatBoostRanker()
            ranker.load_model(j)
            result += ranker.predict(test_chunk[FEATURES]) / 5

        pred['pred'] = result
        pred_all.append(pred)

        del test_chunk
        gc.collect()
        
    # save pred
    pred_all = pd.concat(pred_all)
    pred_all['pred'] = pred_all['pred'].astype(np.float32)
    
    # save
    if type_name != 'click_all':
        pred_all.to_parquet(f'{output_path}{type_name}_test_makotu_v3.parquet')
    else:
        pred_all.to_parquet(f'{output_path}click_test_makotu_v3_all_target.parquet')

# +
candidate_path = '../../input/candidate/'
datamart_path ='../../input/feature/'
oof_read_path = '../../output/'
output_path = '../../output/'
model_path = '../../model_v3/'
feature_dict_path = '../../config/feature_config.yaml'
co_matrix_dict_path = '../../config/co_matrix_config.yaml'
oof_dict_path = '../../config/oof_config.yaml'

for t in ['click', 'click_all', 'cart', 'order']:
#for t in ['cart', 'order']:
    main(t, model_path + t, candidate_path, datamart_path, oof_read_path, output_path,
         feature_dict_path, co_matrix_dict_path, oof_dict_path)
