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
import polars as pl
import yaml
import cudf, itertools
print('We will use RAPIDS version',cudf.__version__)
cudf.set_option("default_integer_bitwidth", 32)
cudf.set_option("default_float_bitwidth", 32)


def make_all_click_data(preprocess_path, raw_opt_path):

    train = cudf.read_parquet(preprocess_path + 'train.parquet')
    test = cudf.read_parquet(preprocess_path + 'test.parquet')
    train = train.sort_values(['session', 'ts'])
    test = test.sort_values(['session', 'ts'])
    merge = cudf.concat([train, test])
    merge = merge.sort_values(['session', 'ts'])
    del train, test

    all_data = cudf.read_parquet(raw_opt_path + 'train.parquet')
    all_data = all_data.sort_values(['session', 'ts'])

    all_data['rank'] = 1
    merge['rank'] = 1
    all_data['rank'] = all_data.groupby('session')['rank'].cumsum()
    merge['rank'] = merge.groupby('session')['rank'].cumsum()
    merge = merge.groupby('session')['rank'].max().reset_index()
    merge.columns = ['session', 'rank_max']
    all_data = all_data.merge(merge, on = 'session', how = 'left')
    all_data = all_data[all_data['rank'] > all_data['rank_max']]

    del merge
    gc.collect()

    all_data = all_data[['session', 'aid']].drop_duplicates()
    all_data['target'] = 1
                        
    return all_data


def make_candidate_row(path_name_dict):
    
    for n, i in enumerate(path_name_dict.keys()):
        
        print(i)
        candidate = cudf.DataFrame(pd.read_parquet(datamart_path + i))
        if 'rank' not in candidate.columns:        
            candidate['rank'] = candidate.groupby(['session'])[path_name_dict[i][0]].rank(ascending = False, method = 'max')
        candidate = candidate[candidate['rank'] <= path_name_dict[i][1]].reset_index(drop=True)
        print(candidate.shape)
        #print(candidate['session'].value_counts().value_counts().head(10))
        
        if n == 0:
            candidate_all = candidate
        else:
            candidate_all = candidate_all[['session', 'aid']].merge(candidate[['session', 'aid']], 
                                                            on = ['session', 'aid'], how = 'outer')
        print('all_candidate:', candidate_all.shape)
        
        del candidate
        gc.collect()
    
    return candidate_all


def calc_recall(gt_df, pred_df):
    #weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
    test_labels = gt_df.merge(pred_df, how='left', on=['session'])
    #test_labels = gt_df.merge(pred_df, how='inner', on=['session'])
    test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
    test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)
    recall = test_labels['hits'].sum() / test_labels['gt_count'].sum()
    #score = weights[t]*recall
    print(f'recall =',recall)
    
    return recall


raw_opt_path = '../../data/raw_opt/'
preprocess_path = '../../data/val_radek/'
datamart_path = '../../github/data/feature_used/'
output_path = '../../github/data/candidate/'


def main(raw_opt_path, preprocess_path, datamart_path, output_path):

    for dict_ in ['order', 'cart', 'click', 'click_all']:

        print(dict_)

        for prefix in ['train_', 'test_']:

            # 0.8082842 3.40億
            order_dict = {
                prefix + 'click_click_allterm_last.parquet': [None, 100],
                prefix + 'click_click_allterm_top.parquet': [None, 20],
                prefix + 'click_click_allterm_hour.parquet': [None, 100],
                prefix + 'click_click_allterm_day.parquet': [None, 30],

                prefix + 'click_buy_allterm_last.parquet': [None, 40],
                prefix + 'click_buy_allterm_top.parquet': [None, 40],
                prefix + 'click_buy_allterm_hour.parquet': [None, 40],
                prefix + 'click_buy_allterm_day.parquet': [None, 10],

                prefix + 'buy_click_allterm_all.parquet': [None, 40],
                prefix + 'buy_buy_allterm_all.parquet': [None, 40],

                prefix + 'click_click_dup_last.parquet': [None, 20],
                prefix + 'click_click_dup_top.parquet': [None, 10],
                prefix + 'click_click_dup_hour.parquet': [None, 20],

                prefix + 'click_buy_dup_last.parquet': [None, 20],
                prefix + 'click_buy_dup_top.parquet': [None, 10],
                prefix + 'click_buy_dup_hour.parquet': [None, 20],
                prefix + 'buy_click_dup_all.parquet': [None, 20],
                prefix + 'buy_buy_dup_all.parquet': [None, 20],

                prefix + 'click_click_dup_wlen_last.parquet': [None, 20],
                prefix + 'click_click_dup_wlen_hour.parquet': [None, 20],
                prefix + 'click_buy_dup_wlen_last.parquet': [None, 20],
                prefix + 'click_buy_dup_wlen_hour.parquet': [None, 20],

                prefix + 'click_click_base_last.parquet': [None, 50],
                prefix + 'click_click_base_top.parquet': [None, 10],
                prefix + 'click_click_base_hour.parquet': [None, 50],
                prefix + 'buy_click_base_all.parquet': [None, 40],
                prefix + 'buy_buy_base_all.parquet': [None, 40],

                prefix + 'click_click_base_wlen_last.parquet': [None, 40],
                prefix + 'click_click_base_wlen_top.parquet': [None, 10],
                prefix + 'click_click_base_wlen_hour.parquet': [None, 30],
                prefix + 'buy_click_base_wlen_all.parquet': [None, 20],
                prefix + 'buy_buy_base_wlen_all.parquet': [None, 20],

                prefix + 'click_click_base_hour_last.parquet': [None, 15],
                prefix + 'click_click_base_hour_hour.parquet': [None, 15],
                prefix + 'click_click_dup_hour_last.parquet': [None, 5],
                prefix + 'click_click_dup_hour_hour.parquet': [None, 5],
            }

            ## 0.678994976 3.40億
            cart_dict = order_dict.copy()

            ## 0.6826327 2.73億
            click_dict = {
                prefix + 'click_click_dup_wlen_last.parquet': [None, 70],
                prefix + 'click_click_dup_wlen_hour.parquet': [None, 50],
                prefix + 'click_click_dup_wlen_day.parquet': [None, 10],

                prefix + 'click_click_base_hour_last.parquet': [None, 90],
                prefix + 'click_click_base_hour_top.parquet': [None, 5],
                prefix + 'click_click_base_hour_hour.parquet': [None, 60],
                prefix + 'click_click_base_hour_day.parquet': [None, 20],

                prefix + 'click_click_dup_hour_last.parquet': [None, 30],
                prefix + 'click_click_dup_hour_hour.parquet': [None, 30],

                prefix + 'click_click_base_last.parquet': [None, 30],
                prefix + 'click_click_base_top.parquet': [None, 5],
                prefix + 'click_click_base_hour.parquet': [None, 30],
                prefix + 'click_click_base_day.parquet': [None, 10],

                prefix + 'click_click_allterm_last.parquet': [None, 30],
                prefix + 'click_click_allterm_top.parquet': [None, 5],
                prefix + 'click_click_allterm_hour.parquet': [None, 30],
                prefix + 'click_click_allterm_day.parquet': [None, 10],

                prefix + 'click_click_dup_last.parquet': [None, 30],
                prefix + 'click_click_dup_top.parquet': [None, 5],
                prefix + 'click_click_dup_hour.parquet': [None, 30],
                prefix + 'click_click_dup_day.parquet': [None, 10],

                prefix + 'click_click_w2v_last_w2v.parquet': [None, 10],
                prefix + 'click_click_w2v_hour_w2v.parquet': [None, 5],
            }

            if prefix == 'test_':
                train = pd.read_parquet(raw_opt_path + 'test.parquet')
            else:
                train = pd.read_parquet(preprocess_path + 'test.parquet')

            hist_all = cudf.DataFrame(train[['session', 'aid']].value_counts().reset_index())

            if dict_ == 'order':
                candidate_all = make_candidate_row(order_dict)
            elif dict_ == 'cart':
                candidate_all = make_candidate_row(cart_dict)
            else:
                candidate_all = make_candidate_row(click_dict)

            gc.collect()

            candidate_all['session'] = candidate_all['session'].astype(np.int32)
            candidate_all['aid'] = candidate_all['aid'].astype(np.int32)
            candidate_all = candidate_all.merge(hist_all[['session', 'aid']], on = ['session', 'aid'], how = 'outer')
            print(candidate_all.shape)
            del hist_all
            gc.collect()

            if prefix == 'train_':
                if dict_ != 'click_all':
                    target = pd.read_parquet(preprocess_path + 'test_labels.parquet')
                    target = target[target['type'] == f'{dict_}s']

                    session_list = []
                    aid_list = []

                    for row in tqdm(target.values):
                        for t in row[2]:
                            session_list.append(row[0])
                            aid_list.append(t)

                    target_df = pd.DataFrame(session_list, columns = ['session'])
                    target_df['aid'] = aid_list
                    target_df['target'] = 1
                    target_df['target'] = target_df['target'].astype(np.int16)
                    target_df = cudf.DataFrame(target_df)

                    candidate_all = candidate_all.merge(target_df, on = ['session', 'aid'], how = 'left')
                    candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
                    candidate_all = candidate_all.to_pandas()
                else:
                    click_all_target = make_all_click_data(preprocess_path, raw_opt_path)
                    candidate_all = candidate_all.merge(click_all_target, on = ['session', 'aid'], how = 'left')
                    candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
                    candidate_all = candidate_all.to_pandas()
                    del click_all_target

            else:
                candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
                candidate_all = candidate_all.to_pandas()

            candidate_all.to_parquet(output_path + prefix + f'{dict_}_candidate.parquet')

            if prefix == 'train_':
                print('calc recall...')
                pred_temp = candidate_all[['session', 'aid']].groupby('session')['aid'].apply(lambda x: " ".join(map(str,x)))
                pred_temp = pred_temp.reset_index()
                pred_temp.columns = ['session', 'labels']

                target['ground_truth'] = target.ground_truth.apply(lambda x: x[0].astype(str).split(' '))
                pred_temp['labels'] = pred_temp[['labels']].apply(lambda x: x.str.split(' '))
                print(calc_recall(target[target['type'] == f'{dict_}s'], pred_temp))


def main(raw_opt_path, dict_, preprocess_path, datamart_path, output_path, calc_recall):

    for prefix in ['train_', 'test_']:

        # 0.8082842 3.40億
        order_dict = {
            prefix + 'click_click_allterm_last.parquet': [None, 100],
            prefix + 'click_click_allterm_top.parquet': [None, 20],
            prefix + 'click_click_allterm_hour.parquet': [None, 100],
            prefix + 'click_click_allterm_day.parquet': [None, 30],

            prefix + 'click_buy_allterm_last.parquet': [None, 40],
            prefix + 'click_buy_allterm_top.parquet': [None, 40],
            prefix + 'click_buy_allterm_hour.parquet': [None, 40],
            prefix + 'click_buy_allterm_day.parquet': [None, 10],

            prefix + 'buy_click_allterm_all.parquet': [None, 40],
            prefix + 'buy_buy_allterm_all.parquet': [None, 40],

            prefix + 'click_click_dup_last.parquet': [None, 20],
            prefix + 'click_click_dup_top.parquet': [None, 10],
            prefix + 'click_click_dup_hour.parquet': [None, 20],

            prefix + 'click_buy_dup_last.parquet': [None, 20],
            prefix + 'click_buy_dup_top.parquet': [None, 10],
            prefix + 'click_buy_dup_hour.parquet': [None, 20],
            prefix + 'buy_click_dup_all.parquet': [None, 20],
            prefix + 'buy_buy_dup_all.parquet': [None, 20],

            prefix + 'click_click_dup_wlen_last.parquet': [None, 20],
            prefix + 'click_click_dup_wlen_hour.parquet': [None, 20],
            prefix + 'click_buy_dup_wlen_last.parquet': [None, 20],
            prefix + 'click_buy_dup_wlen_hour.parquet': [None, 20],

            prefix + 'click_click_base_last.parquet': [None, 50],
            prefix + 'click_click_base_top.parquet': [None, 10],
            prefix + 'click_click_base_hour.parquet': [None, 50],
            prefix + 'buy_click_base_all.parquet': [None, 40],
            prefix + 'buy_buy_base_all.parquet': [None, 40],

            prefix + 'click_click_base_wlen_last.parquet': [None, 40],
            prefix + 'click_click_base_wlen_top.parquet': [None, 10],
            prefix + 'click_click_base_wlen_hour.parquet': [None, 30],
            prefix + 'buy_click_base_wlen_all.parquet': [None, 20],
            prefix + 'buy_buy_base_wlen_all.parquet': [None, 20],

            prefix + 'click_click_base_hour_last.parquet': [None, 15],
            prefix + 'click_click_base_hour_hour.parquet': [None, 15],
            prefix + 'click_click_dup_hour_last.parquet': [None, 5],
            prefix + 'click_click_dup_hour_hour.parquet': [None, 5],
        }

        ## 0.678994976 3.40億
        cart_dict = order_dict.copy()

        ## 0.6826327 2.73億
        click_dict = {
            prefix + 'click_click_dup_wlen_last.parquet': [None, 70],
            prefix + 'click_click_dup_wlen_hour.parquet': [None, 50],
            prefix + 'click_click_dup_wlen_day.parquet': [None, 10],

            prefix + 'click_click_base_hour_last.parquet': [None, 90],
            prefix + 'click_click_base_hour_top.parquet': [None, 5],
            prefix + 'click_click_base_hour_hour.parquet': [None, 60],
            prefix + 'click_click_base_hour_day.parquet': [None, 20],

            prefix + 'click_click_dup_hour_last.parquet': [None, 30],
            prefix + 'click_click_dup_hour_hour.parquet': [None, 30],

            prefix + 'click_click_base_last.parquet': [None, 30],
            prefix + 'click_click_base_top.parquet': [None, 5],
            prefix + 'click_click_base_hour.parquet': [None, 30],
            prefix + 'click_click_base_day.parquet': [None, 10],

            prefix + 'click_click_allterm_last.parquet': [None, 30],
            prefix + 'click_click_allterm_top.parquet': [None, 5],
            prefix + 'click_click_allterm_hour.parquet': [None, 30],
            prefix + 'click_click_allterm_day.parquet': [None, 10],

            prefix + 'click_click_dup_last.parquet': [None, 30],
            prefix + 'click_click_dup_top.parquet': [None, 5],
            prefix + 'click_click_dup_hour.parquet': [None, 30],
            prefix + 'click_click_dup_day.parquet': [None, 10],

            prefix + 'click_click_w2v_last_w2v.parquet': [None, 10],
            prefix + 'click_click_w2v_hour_w2v.parquet': [None, 5],
        }

        if prefix == 'test_':
            train = pd.read_parquet(raw_opt_path + 'test.parquet')
        else:
            train = pd.read_parquet(preprocess_path + 'test.parquet')

        hist_all = cudf.DataFrame(train[['session', 'aid']].value_counts().reset_index())

        if dict_ == 'order':
            candidate_all = make_candidate_row(order_dict)
        elif dict_ == 'cart':
            candidate_all = make_candidate_row(cart_dict)
        else:
            candidate_all = make_candidate_row(click_dict)

        gc.collect()

        candidate_all['session'] = candidate_all['session'].astype(np.int32)
        candidate_all['aid'] = candidate_all['aid'].astype(np.int32)
        candidate_all = candidate_all.merge(hist_all[['session', 'aid']], on = ['session', 'aid'], how = 'outer')
        print(candidate_all.shape)
        del hist_all
        gc.collect()

        if prefix == 'train_':
            if dict_ != 'click_all':
                target = pd.read_parquet(preprocess_path + 'test_labels.parquet')
                target = target[target['type'] == f'{dict_}s']

                session_list = []
                aid_list = []

                for row in tqdm(target.values):
                    for t in row[2]:
                        session_list.append(row[0])
                        aid_list.append(t)

                target_df = pd.DataFrame(session_list, columns = ['session'])
                target_df['aid'] = aid_list
                target_df['target'] = 1
                target_df['target'] = target_df['target'].astype(np.int16)
                target_df = cudf.DataFrame(target_df)

                candidate_all = candidate_all.merge(target_df, on = ['session', 'aid'], how = 'left')
                candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
                candidate_all = candidate_all.to_pandas()
            else:
                click_all_target = make_all_click_data(preprocess_path, raw_opt_path)
                candidate_all = candidate_all.merge(click_all_target, on = ['session', 'aid'], how = 'left')
                candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
                candidate_all = candidate_all.to_pandas()
                del click_all_target

        else:
            candidate_all = candidate_all.sort_values(['session', 'aid']).reset_index(drop=True)
            candidate_all = candidate_all.to_pandas()

        candidate_all.to_parquet(output_path + prefix + f'{dict_}_candidate.parquet')

        if calc_recall == True:
            print('calc recall...')
            pred_temp = candidate_all[['session', 'aid']].groupby('session')['aid'].apply(lambda x: " ".join(map(str,x)))
            pred_temp = pred_temp.reset_index()
            pred_temp.columns = ['session', 'labels']

            target['ground_truth'] = target.ground_truth.apply(lambda x: x[0].astype(str).split(' '))
            pred_temp['labels'] = pred_temp[['labels']].apply(lambda x: x.str.split(' '))
            print(calc_recall(target[target['type'] == f'{dict_}s'], pred_temp))

        gc.collect()


# path
raw_opt_path = '../../input/train_test/'
preprocess_path = '../../input/train_valid/'
datamart_path = '../../input/feature/'
output_path = '../../input/candidate/'
calc_recall = False

main(raw_opt_path, 'order', preprocess_path, datamart_path, output_path, calc_recall)
main(raw_opt_path, 'cart', preprocess_path, datamart_path, output_path, calc_recall)
main(raw_opt_path, 'click', preprocess_path, datamart_path, output_path, calc_recall)
main(raw_opt_path, 'click_all', preprocess_path, datamart_path, output_path, calc_recall)










