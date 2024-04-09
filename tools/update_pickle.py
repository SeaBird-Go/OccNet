'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-04-01 23:22:54
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import mmengine
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


def check_pickle_contents():
    pkl_fp = "data/lightwheelocc/lightwheel_occ_infos_train.pkl"
    data = mmengine.load(pkl_fp)
    print(data.keys())

    data_infos = data['infos']
    print(len(data_infos))

    pkl_fp2 = "data/nuscenes/nuscenes_infos_train_occ.pkl"
    data2 = mmengine.load(pkl_fp2)
    data_infos2 = data2['infos']
    print(len(data_infos2))


def update_lightwheel_pkl_with_absolute_occ_path():
    for split in ['train', 'val']:
        print("Processing split: ", split)
        pkl_fp = f"data/lightwheelocc/lightwheel_occ_infos_{split}.pkl"
        data = mmengine.load(pkl_fp)
        print(data.keys())

        data_infos = data['infos']
        print(len(data_infos))

        for info in tqdm(data_infos):
            occ_path = info['occ_path']
            occ_path_new = osp.join("data/lightwheelocc", occ_path)
            info['occ_path'] = occ_path_new
        
        mmengine.dump(data, pkl_fp)

    
if __name__ == "__main__":
    update_lightwheel_pkl_with_absolute_occ_path()
