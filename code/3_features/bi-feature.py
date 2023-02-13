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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import os
import gc
from tqdm.notebook import tqdm
import polars as pl

# +
USER_ID = 'session'
ITEM_ID = 'aid'

CLICK = 0
CART = 1
ORDER = 2
# -

raw_opt_path = '../../input/train_test/'
preprocess_path = '../../input/train_valid/'
datamart_path = '../../input/feature/'
candidate_path = '../../input/candidate/'
sampling = True


def get_bigram_fea(train_sample, user_2_test_action_with_type, bigram_count, name):
    feas = {}
    for src in [CLICK, CART]:
        if src == 0:
            src_name = 'click'
        else:
            src_name = 'cart'
        sum_sim, mean_sim, max_sim, min_sim, last_sim = [], [], [], [], []
        last_sim2, last_sim3 = [], []
        for idx, u, i in tqdm(train_sample[[USER_ID, ITEM_ID]].itertuples(), total=len(train_sample)):
            acts = user_2_test_action_with_type[src].get(u, [])
            sims = []
            if len(acts)>0:
                for a in acts:
                    sims.append(bigram_count.get((a, i), 0))
                sum_sim.append(np.sum(sims))
                mean_sim.append(np.mean(sims))
                max_sim.append(np.max(sims))
                min_sim.append(np.min(sims))
                last_sim.append(sims[-1])
            else:
                sum_sim.append(-1)
                mean_sim.append(-1)
                max_sim.append(-1)
                min_sim.append(-1)
                last_sim.append(-1)
        feas.update({
            name + f'_{src_name}_sum': sum_sim, name + f'_{src_name}_mean': mean_sim, name + f'_{src_name}_max': max_sim, name + f'_{src_name}_min': min_sim,
            name + f'_{src_name}_last': last_sim
        })
    return pd.DataFrame(feas)


def make_bi_feature(prefix, raw_opt_path, preprocess_path, datamart_path, sampling):
    
    if prefix == 'test_':
        train_actions = pd.read_parquet(raw_opt_path + 'train.parquet')
        test_actions = pd.read_parquet(raw_opt_path + 'test.parquet')
    else:
        train_actions = pd.read_parquet(preprocess_path + 'train.parquet')
        test_actions = pd.read_parquet(preprocess_path + 'test.parquet')

    if sampling == True:
        train_actions = train_actions.head(1000)
        test_actions = test_actions.head(1000)

    # calc item frequency, used to normalize the covisit （very important）
    df = pd.concat([train_actions, test_actions])
    item_cnt = {'all': df.groupby(ITEM_ID).size().to_dict()}
    for t in [CLICK, CART, ORDER]:
        item_cnt[t] = df.loc[df['type'] == t].groupby(ITEM_ID).size().to_dict()
        
    train_pairs = pd.concat([train_actions, test_actions])
    train_pairs['aid_next'] = train_pairs.groupby('session')[ITEM_ID].shift(-1)
    train_pairs = train_pairs.dropna()
    train_pairs['aid_next'] = train_pairs['aid_next'].astype('int32')
    train_pairs = train_pairs[['aid', 'aid_next']]
    bigram_counter = train_pairs.groupby(['aid', 'aid_next']).size().to_dict()
    normed_bigram_counter = {}
    
    print('start bigram counter...')
    for (a1, a2), cnt in bigram_counter.items():
        normed_bigram_counter[(a1, a2)] = cnt/np.sqrt(item_cnt['all'].get(a1, 1) * item_cnt['all'].get(a2, 1))
    
    if prefix == 'train_':
        train_sample = pd.read_parquet(candidate_path + 'train_order_candidate.parquet')
    else:
        train_sample = pd.read_parquet(candidate_path + 'test_order_candidate.parquet')
    
    if sampling == True:
        train_sample = train_sample.head(100000)
        
    user_2_test_action_with_type = {t: test_actions.loc[test_actions['type'] == t, :].groupby(USER_ID)[ITEM_ID].agg(list) for t in [CLICK, CART, ORDER]}
    normed_bigram = get_bigram_fea(train_sample, user_2_test_action_with_type, normed_bigram_counter, 'bigram_normed')
    
    bigram = pl.concat([pl.DataFrame(train_sample[['session', 'aid']]), pl.DataFrame(normed_bigram)], how="horizontal")
    bigram = bigram.with_column(pl.col(['session', 'aid']).cast(pl.Int32, strict=False))
    bigram = bigram.with_column(pl.col(['bigram_normed_click_sum', 'bigram_normed_click_mean',
           'bigram_normed_click_max', 'bigram_normed_click_min',
           'bigram_normed_click_last', 'bigram_normed_cart_sum',
           'bigram_normed_cart_mean', 'bigram_normed_cart_max',
           'bigram_normed_cart_min', 'bigram_normed_cart_last']).cast(pl.Float32, strict=False))
    
    bigram.to_pandas().to_parquet(datamart_path + prefix + 'bigram_feature.parquet')
    gc.collect()


make_bi_feature('test_', raw_opt_path, preprocess_path, datamart_path, False)
make_bi_feature('train_', raw_opt_path, preprocess_path, datamart_path, False)






