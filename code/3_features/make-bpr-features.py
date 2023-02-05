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

import pickle
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# ### data load

def user_item_dot(user_id, item_id, u2emb, i2emb):
    default_emb = np.zeros(64+1)
    u_mat = np.stack([u2emb.get(u, default_emb) for u in user_id])
    i_mat = np.stack([i2emb.get(i, default_emb) for i in item_id])
    return np.sum(u_mat * i_mat, axis=1)


def main(bpr_path, candidate_path, datamart_path):
    
    for prefix in ['train_', 'test_']:

        print(prefix)
        train_or_test = prefix.split('_')[0]

        with open(f'{bpr_path}u2emb_{train_or_test}.pkl', 'rb') as fp:
            u2emb = pickle.load(fp)

        with open(f'{bpr_path}i2emb_{train_or_test}.pkl', 'rb') as fp:
            i2emb = pickle.load(fp)

        for candidate_file_name in ['click_candidate.parquet', 'cart_candidate.parquet', 'order_candidate.parquet']:

            print(candidate_file_name)
            type_name = candidate_file_name.split('_')[0] + '_'

            train_sample = pd.read_parquet(candidate_path + prefix + candidate_file_name)

            name = 'bpr'
            chunk_size = 5000000
            chunk_cnt = len(train_sample) // chunk_size

            pred = np.concatenate([
                user_item_dot(
                    train_sample['session'].iloc[(c * chunk_size):((c + 1) * chunk_size)],
                    train_sample['aid'].iloc[(c * chunk_size):((c + 1) * chunk_size)],
                    u2emb, i2emb
                ) for c in tqdm(range(chunk_cnt+1))
            ])
            train_sample[name] = pred

            train_sample.to_parquet(f'{datamart_path}{prefix}{type_name}_bpr_feature.parquet')


candidate_path = '../../input/candidate/'
datamart_path = '../../input/feature/'
bpr_path = '../../input/preprocess/'

main(bpr_path, candidate_path, datamart_path)








