import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import socket
import warnings
import itertools
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from pathlib import Path

from expl_perf_drop.utils import EarlyStopping, has_checkpoint, load_checkpoint, save_checkpoint, Tee, to_device, print_row
from expl_perf_drop.data import select as data_select
from expl_perf_drop.models import select as model_select
from expl_perf_drop.metrics import select as metric_select
from expl_perf_drop.explainers import CGExplainerDR
from expl_perf_drop.SHAPExplainer import SHAPExplainer


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help = 'Experiment name for downstream tracking purposes')
parser.add_argument('--output_dir', type=str, required = True)
parser.add_argument('--model_dir', type=str, required = True)
parser.add_argument('--metric', type = str, required = True, choices = ['mse', 'acc', 'auroc', 'brier'])
parser.add_argument('--method', type = str, choices = ['ours', 'shap'], required = True)
parser.add_argument('--dataset', type = str, choices = ['synthetic', 'celebA', 'cmnist', 'camelyon'], required = True)
parser.add_argument('--graph', type = str, choices = ["normal", 'marginals', 'conds', 'topo'], default = "normal") 
parser.add_argument('--target_data_dir', type = str, default = '/data/healthy-ml/scratch/haoran/results/expl_perf_drop/data')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_seed', type=int, default=42)
parser.add_argument('--test_pct', type=float, default = 0.25)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--weight_model', type = str, choices = ['xgb', 'lr'], default = 'xgb')
parser.add_argument('--imp_weight_type', type = str, choices = ["normal", 'self_normalize'], default = "normal") 
parser.add_argument('--clip_weights', action = 'store_true')
parser.add_argument('--clip_weight_thres', type=float, default = None, help = 'importance weight clip threshold, should be >= 1')
parser.add_argument('--clip_probs', action = 'store_true')
parser.add_argument('--clip_prob_thres', type=float, default = None, help = 'weight model prediction probability clip threshold, should be >= 0.5')
parser.add_argument('--shapley_method', type = str, choices = ["debug", "AUTO", "EXACT", "EXACT_FAST", "EARLY_STOPPING"], default = "AUTO") 
parser.add_argument('--calibrate_weight_models', action = 'store_true')
parser.add_argument('--calc_inv_expl', action = 'store_true')

# only for spurious synthetic (target dataset)
parser.add_argument('--spu_q', type=float, default = 0.5)
parser.add_argument('--spu_n', type=int, default = 20000)
parser.add_argument('--spu_y_noise', type=float, default = 0.25)
parser.add_argument('--spu_mu_add', type=float, default = 3.)
parser.add_argument('--spu_x1_weight', type=float, default = 1.)

# only for wilds datasets (target dataset)
parser.add_argument('--oversample_ratio', type=float, default = 1., help = 'how much to upsample/downsample minority/majority groups')

# cmnist
parser.add_argument('--cmnist_label_prob', type=float, default = 0.5)
parser.add_argument('--cmnist_color_prob', type=float, default = 0.15)
parser.add_argument('--cmnist_flip_prob', type=float, default = 0.25)

# camelyon
parser.add_argument('--camelyon_target_site', type=int, default = 0, choices = [0, 1, 2, 3, 4])
parser.add_argument('--camelyon_graph_direction', choices = ['causal', 'anticausal'], default = 'causal')

args = parser.parse_args()
hparams = vars(args)

os.makedirs(args.output_dir, exist_ok=True)

if not args.debug:
    sys.stdout = Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output_dir, 'err.txt'))

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tNode: {}".format(socket.gethostname()))

print('Args:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_dir = Path(args.model_dir)
model_train_hparams = json.load((model_dir/'args.json').open('r'))
model = pickle.load((model_dir/'model.pkl').open('rb'))

hparams['model_train_hparams'] = model_train_hparams
assert model_train_hparams['dataset'] == args.dataset

with open(os.path.join(args.output_dir,'args.json'), 'w') as f:
    json.dump(hparams, f, indent = 4)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

metric = metric_select(args.metric)

model_train_hparams['camelyon_graph_direction'] = hparams['camelyon_graph_direction']
data = data_select(model_train_hparams, device)
source_train_df, source_eval_df = data.get_source_train_test()
target_train_df, target_eval_df = data.get_target_train_test(hparams)

cont_model = model_select(args.weight_model, 'regression')
dis_model = model_select(args.weight_model, 'classification')

clip_weight_thres = args.clip_weight_thres if args.clip_weights else None
clip_prob_thres = args.clip_prob_thres if args.clip_probs else None

if args.graph == 'normal':
    graph = data.GRAPH
elif args.graph == 'marginals':
    graph = data.GRAPH.to_all_marginals()
elif args.graph == 'conds':
    graph = data.GRAPH.to_all_conds()
elif args.graph == 'topo':
    graph = data.GRAPH.to_topo()

if args.method == 'ours':
    exp = CGExplainerDR(graph, source_train_df, source_eval_df, target_train_df, target_eval_df,
            data.TRAIN_FEATURES, data.VAR_CATEGORIES, data.TARGET_NAME, dis_model, cont_model, calibrate_weight_models = args.calibrate_weight_models,
            clip_weight_thres = clip_weight_thres, clip_prob_thres = clip_prob_thres, imp_weight_type = args.imp_weight_type,
            shapley_method = args.shapley_method)

    perfs = exp.get_perf_on_sets(model, metric)
    res = exp.explain(model, metric)
    scaled_res = exp.scale(res, perfs)
    
    final = {
        'perfs': perfs,
        'expl': res,
        'scaled_expl': scaled_res        
    }

    if args.calc_inv_expl:
        inv_exp = CGExplainerDR(graph, target_train_df, target_eval_df, source_train_df, source_eval_df, # swap source and target
                data.TRAIN_FEATURES, data.VAR_CATEGORIES, data.TARGET_NAME, dis_model, cont_model, calibrate_weight_models = args.calibrate_weight_models,
                clip_weight_thres = clip_weight_thres, clip_prob_thres = clip_prob_thres, imp_weight_type = args.imp_weight_type,
                shapley_method = args.shapley_method)

        inv_res = inv_exp.explain(model, metric)
        inv_scaled_res = inv_exp.scale(res, inv_exp.get_perf_on_sets(model, metric))

        final = {**final, **{
            'inv_expl': -inv_res,
            'inv_scaled_expl': -inv_scaled_res,
            'avg_expl': (res - inv_res)/2,
            'avg_scaled_expl': (scaled_res - inv_scaled_res)/2
        }}
    
    pickle.dump({
        'cache': exp.cache,
        'weight_model_accs': exp.weight_model_accs
    },open(os.path.join(args.output_dir,'cache.pkl'), 'wb'))

elif args.method == 'shap':
    exp = SHAPExplainer(graph, source_train_df, source_eval_df, target_train_df, target_eval_df,
             data.TRAIN_FEATURES, data.VAR_CATEGORIES, data.TARGET_NAME)
    perfs = exp.get_perf_on_sets(model, metric)
    res = exp.explain(model, metric)

    final = {
        'perfs': perfs,
        'expl': res,  
    }

pickle.dump(final, open(os.path.join(args.output_dir,'stats.pkl'), 'wb'))

with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    f.write('done')
