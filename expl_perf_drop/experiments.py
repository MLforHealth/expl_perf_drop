import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import json
from tqdm import tqdm
import pickle
import copy

def combinations_base(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def combinations(grid):
    sub_exp_names = set()
    args = []
    for i in grid:
        if isinstance(grid[i], dict):
            for j in grid[i]:
                sub_exp_names.add(j)
    if len(sub_exp_names) == 0:
        return combinations_base(grid)

    for i in grid:
        if isinstance(grid[i], dict):
            assert set(list(grid[i].keys())) == sub_exp_names, f'{i} does not have all sub exps ({sub_exp_names})' 
    for n in sub_exp_names:
        sub_grid = grid.copy()
        for i in sub_grid:
            if isinstance(sub_grid[i], dict):
                sub_grid[i] = sub_grid[i][n]
        args += combinations_base(sub_grid)
    return args
        
def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment]().get_hparams()    

def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname


datasets = ['synthetic', 'celebA', 'cmnist', 'camelyon']
model_dir = Path('/home/gridsan/hrzhang/results/expl_perf_drop/models')

class train_model():
    fname = 'train_model'
    def __init__(self):
        self.hparams1 = {
            'exp_name': ['train_model'],
            'dataset': ['celebA'],
            'model': ['lr'],
            'emb_model': ['resnet18']
        }

        self.hparams2 = {
            'exp_name': ['train_model'],
            'dataset': ['synthetic'],
            'model': ['lr', 'xgb']
        }

        self.hparams3 = {
            'exp_name': ['train_model'],
            'dataset': ['cmnist'],
            'model': ['nn', 'gdro_nn']
        }

        self.hparams4 = {
            'exp_name': ['train_model'],
            'dataset': ['camelyon'],
            'model': ['lr'],
            'emb_model': ['resnet18'],
            'camelyon_source_site': [0, 1, 2, 3, 4]
        }

    def get_hparams(self):
        return combinations(self.hparams1) + combinations(self.hparams2) + combinations(self.hparams3) + combinations(self.hparams4) 

class explain():
    fname = 'explain'
    def __init__(self):        
        models = {i: [] for i in datasets}

        for i in model_dir.glob('**/done'):
            args = json.load((i.parent/'args.json').open('rb'))
            if args['dataset'] not in datasets:
                continue
            if args['exp_name'] == 'train_model':
                models[args['dataset']].append(str(i.parent))

        self.base_hparams = {
            'exp_name': ['explain'],
            'metric': ['acc',  'brier'],
            'weight_model': ['xgb'],            
        }

        self.syn = {
            'dataset': ['synthetic'],
            'graph':{
                'exp1': ["normal", 'marginals', 'conds', 'topo'],
                'exp2': ['normal']
            },
            'method':{
                'exp1': ['ours'],
                'exp2': ['shap']
            },
            'model_dir': models['synthetic'],
        }

        self.cmnist = {
            'dataset': ['cmnist'],
            'graph':{
                'exp1': ["normal", 'marginals', 'conds', 'topo'],
            },
            'method':{
                'exp1': ['ours'],
            },
            'model_dir': models['cmnist'],
        }

        self.synthetic_hparams = {            
            'spu_q': np.concatenate((np.linspace(0, 1, 11), np.array([0.05, 0.95]))),
            'spu_mu_add': np.arange(-1, 6, 1),
            'spu_y_noise': [0.25]            
        }

        self.synthetic_hparams2 = {
            'spu_q': np.concatenate((np.linspace(0, 1, 11), np.array([0.05, 0.95]))),
            'spu_mu_add': [3.],
            'spu_y_noise': [0.25],
            'spu_x1_weight': [0.]
        }

        self.cmnist_hparams = {
            'cmnist_label_prob': [0.5],
            'cmnist_color_prob': np.concatenate((np.linspace(0, 1, 11), np.array([0.05, 0.95]))),
            'cmnist_flip_prob': [0.25],
        }

        self.cmnist_hparams2 = {
            'cmnist_label_prob': [0.5],
            'cmnist_color_prob': [0.15, 0.5],
            'cmnist_flip_prob': np.linspace(0.05, 0.95, 10),
        }

        self.cmnist_hparams3 = {
            'cmnist_label_prob': np.linspace(0.1, 0.9, 9),
            'cmnist_color_prob': [0.15],
            'cmnist_flip_prob': [0.25],
        }

        self.no_clipping = {
            'clip_probs': [False]
        }

        self.clip_probs = {
            'clip_probs': [True],
            'clip_weights': [False],
            'clip_prob_thres': [0.95, 0.99]
        }
            
    def get_hparams(self):
        return (
        combinations({**self.base_hparams, **self.syn, **self.synthetic_hparams, **self.no_clipping}) + 
        combinations({**self.base_hparams, **self.syn, **self.synthetic_hparams2, **self.no_clipping}) +
        combinations({**self.base_hparams, **self.cmnist, **self.cmnist_hparams, **self.no_clipping}) + 
        combinations({**self.base_hparams, **self.cmnist, **self.cmnist_hparams2, **self.no_clipping}) +
        combinations({**self.base_hparams, **self.cmnist, **self.cmnist_hparams3, **self.no_clipping}) 
        )

class explain_celebA_camelyon():
    fname = 'explain'
    def __init__(self):        
        models = {i: [] for i in datasets}

        for i in model_dir.glob('**/done'):
            args = json.load((i.parent/'args.json').open('rb'))
            if args['dataset'] not in datasets:
                continue
            if args['exp_name'] == 'train_model':
                models[args['dataset']].append(str(i.parent))

        self.base_hparams = {
            'exp_name': ['explain'],
            'metric': ['acc', 'brier'],
            'weight_model': ['xgb'],
        }
        
        celebA_target_dirs = []
        for i in (model_dir.parent/'data'/'celebA').glob('**/labels.csv'):
            if i.parent.name != 'base':
                celebA_target_dirs.append(str(i.parent.absolute()))

        self.celebA_hparams = {
            'dataset': ['celebA'],
            'model_dir': models['celebA'],
            'target_data_dir': celebA_target_dirs,
            'shapley_method': ['EXACT'],
            'graph':{
                'exp1': ["normal", 'marginals', 'conds', 'topo']
            },
            'method':{
                'exp1': ['ours']
            }
        }

        self.camelyon_hparams = {
            'dataset': ['camelyon'],            
            'model_dir': models['camelyon'],
            'shapley_method': ['EXACT'],
            'camelyon_target_site': [0, 1, 2, 3, 4],
            'camelyon_graph_direction': ['causal', 'anticausal'],
            'graph':{
                'exp1': ["normal", "marginals"]
            },
            'method':{
                'exp1': ['ours']
            }
        }

    def get_hparams(self):
        return (
        combinations({**self.base_hparams, **self.celebA_hparams}) + 
        combinations({**self.base_hparams, **self.camelyon_hparams})
        )

class explain_janzing():
    fname = 'explain_janzing'
    def __init__(self):        
        models = {i: [] for i in datasets}

        for i in model_dir.glob('**/done'):
            args = json.load((i.parent/'args.json').open('rb'))
            if args['dataset'] not in datasets:
                continue
            if args['exp_name'] == 'train_model':
                models[args['dataset']].append(str(i.parent))

        self.base_hparams = {
            'exp_name': ['explain_janzing'],
            'weight_model': ['xgb']
        }

        self.synthetic_hparams = {
            'dataset': ['synthetic'],
            'spu_q': np.linspace(0, 1, 11),
            'spu_mu_add': np.arange(-1, 6, 1),
            'spu_y_noise': [0.25],
            'model_dir': [models['synthetic'][0]],
        }

        self.synthetic_hparams2 = {
            'dataset': ['synthetic'],
            'spu_q': np.linspace(0, 1, 11),
            'spu_mu_add': [3.],
            'spu_y_noise': [0.25],
            'spu_x1_weight': [0.],
            'model_dir': [models['synthetic'][0]],
        }


    def get_hparams(self):
        return (
        combinations({**self.base_hparams, **self.synthetic_hparams}) + 
        combinations({**self.base_hparams, **self.synthetic_hparams2}) 
        )

