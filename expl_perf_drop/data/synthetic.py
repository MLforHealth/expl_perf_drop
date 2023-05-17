import numpy as np
from expl_perf_drop.utils import flatten
from expl_perf_drop.utils import Graph
import pandas as pd
from sklearn.model_selection import train_test_split

class BackdoorSpurious():
    TARGET_NAME = 'Y'
    VAR_CATEGORIES = {
        'G': ['G'],
        'X1': ['X1'],
        'X2': ['X2'],
        'X3': ['X3'],
        'Y': ['Y']
    }
    TRAIN_FEATURES = ['X1', 'X2', 'X3']
    TASK_TYPE = 'classification'
    GRAPH = Graph(
        nodes= list(VAR_CATEGORIES.keys()),
        edges=[
            ('G', 'Y'),
            ('G', 'X2'),
            ('G', 'X3'),
            ('Y', 'X1'), 
            ('Y', 'X2'), 
            ('Y', 'X3'), 
        ]
    )
    def __init__(self, hparams):
        self.hparams = hparams 
        self.data_seed = hparams['data_seed']
        self.n = self.hparams['n']
        self.test_pct = self.hparams['test_pct']
        rng = np.random.RandomState(self.data_seed)
        self.w = rng.normal(size = (3, 1))

    def data_to_df(self, X, G, Y):
        df = pd.DataFrame({'G': G.astype(int), 'Y': Y.astype(int)})
        for i in range(1, X.shape[1] + 1):
            df[f'X{i}'] = X[:, i-1]
        return df

    def generate(self, n, rng, q = 0.9, y_noise = 0.25, mu_add = 3, x1_weight = 1.0):
        G = rng.random(size = (n, 1)) >= 0.5
        Y = np.logical_xor(G, rng.random(size = (n, 1)) >= q)     
        Y_noised_1 = np.logical_xor(Y, rng.random(size = (n, 1)) <= y_noise)
        Y_noised_2 = np.logical_xor(Y, rng.random(size = (n, 1)) <= y_noise)   
        Y_noised_3 = np.logical_xor(Y, rng.random(size = (n, 1)) <= y_noise)   
        
        X1 = rng.normal(loc = x1_weight * Y_noised_1, size = (n, 1))
        X2 = rng.normal(loc = Y_noised_2 + G, size = (n, 1))
        X3 = rng.normal(loc = Y_noised_3 + mu_add * G, size = (n, 1))
        X = np.concatenate([X1, X2, X3], axis = -1)

        return self.data_to_df(X, G.squeeze(), Y.squeeze())

    def get_source_train_test(self):
        rng = np.random.RandomState(self.data_seed)
        df = self.generate(self.n, rng, self.hparams['spu_q'], self.hparams['spu_y_noise'], self.hparams['spu_mu_add'], self.hparams['spu_x1_weight'])
        return train_test_split(df, random_state = self.data_seed, shuffle = True, test_size = self.test_pct)

    def get_target_train_test(self, shift_hparams):
        assert shift_hparams['data_seed'] != self.hparams['data_seed']
        rng = np.random.RandomState(shift_hparams['data_seed'])
        df = self.generate(shift_hparams['spu_n'], rng, shift_hparams['spu_q'], shift_hparams['spu_y_noise'], shift_hparams['spu_mu_add'], shift_hparams['spu_x1_weight'])
        return train_test_split(df, random_state = shift_hparams['data_seed'], shuffle = True, test_size = shift_hparams['test_pct'])
