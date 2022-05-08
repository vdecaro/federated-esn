import os
import pickle
import pathlib
from collections import OrderedDict

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np

HHAR_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'raw', 'HHAR')

USERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
PHONES = ['nexus4', 's3', 's3mini', 'samsungold']
WATCHES = ['gear', 'lgwatch']
LABELS = {'stand': 0, 'sit': 1, 'walk': 2, 'stairsup': 3, 'stairsdown': 4, 'bike': 5}
FREQ = {'nexus4': 200, 's3': 150, 's3mini': 100, 'samsungold': 50, 'gear': 100, 'lgwatch': 200}
TARGET_FREQ = 32
TOLERANCE = 5


class HHARDataset(torch.utils.data.Dataset):

    def __init__(self, idx: int, sequence_length: int) -> None:
        super().__init__()
        self.user = USERS[idx]
        self.seq_length = sequence_length

        u_path = os.path.join(HHAR_PATH, 'users', f'{self.user}.pkl')
        if not os.path.exists(u_path):
            self.user_data = self.preprocess()
        else:
            self.user_data = pickle.load(open(u_path, 'rb'))

    def __len__(self):
        return sum([self.user_data[k]['size'] for k in self.user_data])

    def __getitem__(self, i):
        for k in self.user_data:
            dev = self.user_data[k]
            if dev['first_index'] <= i < dev['first_index'] + dev['first_index'] + dev['size']:
                break
        
        i -= dev['first_index']
        size = dev['size']
        if i + self.seq_length >= size:
            i = size - self.seq_length

        return dev['X'][i:i+self.seq_length], dev['Y'][i:i+self.seq_length]

    def preprocess(self):

        rdfs = {
            'p_acc': pd.read_csv(os.path.join(HHAR_PATH, 'Phones_accelerometer.csv')).dropna(subset=['gt']),
            'p_gyr': pd.read_csv(os.path.join(HHAR_PATH, 'Phones_gyroscope.csv')).dropna(subset=['gt']),
            'w_acc': pd.read_csv(os.path.join(HHAR_PATH, 'Watch_accelerometer.csv')).dropna(subset=['gt']),
            'w_gyr': pd.read_csv(os.path.join(HHAR_PATH, 'Watch_gyroscope.csv')).dropna(subset=['gt'])
        }
        u_dict = OrderedDict()
        curr_idx = 0
        for device in PHONES + WATCHES:
            dev = {'1': None, '2': None}
            for i in ['1', '2']:
                dev[i] = self._merge_user_device(
                    rdfs[f"{'p' if device in PHONES else 'w'}_acc"],
                    rdfs[f"{'p' if device in PHONES else 'w'}_gyr"],
                    self.user, 
                    f"{device}_{i}",
                    TOLERANCE
                )
            dev_1_perc, dev_2_perc = dev['1'].isna().sum().max() / len(dev['1']), dev['2'].isna().sum().max() / len(dev['2'])
            chosen_dev, perc = ('1', dev_1_perc) if (dev_1_perc <= dev_2_perc) else ('2', dev_2_perc)
            if perc < 0.1:
                df_dev = dev[chosen_dev]
                down_idx = np.around(np.arange(0, len(df_dev)-1, FREQ[device]/TARGET_FREQ))
                df_dev = df_dev.iloc[down_idx]
                df_dev = df_dev.dropna()
                values = ['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']
                df_dev[values] = (df_dev[values] - df_dev[values].mean()) / df_dev[values].std()
                df_dev['gt'] = df_dev['gt'].apply(lambda x: LABELS[x])
                
                u_dict[device] = {
                    'X': torch.tensor(df_dev[values].values),
                    'Y': torch.tensor(df_dev['gt'].values),
                    'first_index': curr_idx,
                    'size': len(df_dev)
                }
                curr_idx += len(df_dev)

        pickle.dump(u_dict, open(os.path.join(HHAR_PATH, 'users', f'{self.user}.pkl'), 'wb+'))
        
        return u_dict
    
    def _merge_user_device(self, acc_df, gyr_df, user, device, tolerance):
    
        def aux(df, user, device, sd):
            return df.loc[((df['User'] == user) & (df['Device'] == device)), ["Arrival_Time", "x", "y", "z", "gt"]] \
                .sort_values('Arrival_Time') \
                .rename(columns={"x": f"x_{sd}", "y": f"y_{sd}", "z": f"z_{sd}"})
        
        ud_acc, ud_gyr = aux(acc_df, user, device, f'acc'), aux(gyr_df, user, device, f'gyr')
        no_gt = ['Arrival_Time', 'x_gyr', 'y_gyr', 'z_gyr']
        lu_df = pd.merge_asof(ud_acc, ud_gyr[no_gt], on='Arrival_Time', direction='nearest', tolerance=tolerance)
        no_gt = ['Arrival_Time', 'x_acc', 'y_acc', 'z_acc']
        ru_df = pd.merge_asof(ud_gyr, ud_acc[no_gt], on='Arrival_Time', direction='nearest', tolerance=tolerance)
        u_df = pd.concat([lu_df, ru_df[ru_df['x_acc'].isnull()]]).sort_values('Arrival_Time')
        u_df = u_df[['Arrival_Time', 'x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'gt']]
        return u_df
    