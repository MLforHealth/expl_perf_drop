from dowhy.gcm.distribution_change import estimate_distribution_change_scores

from dowhy import gcm
from dowhy.gcm.ml.classification import SklearnClassificationModel
from dowhy.gcm.ml.regression import SklearnRegressionModel

import numpy as np
import pandas as pd
import sys
import argparse
import os
import random
import torch
from pathlib import Path
import socket
import pickle
import json

from expl_perf_drop.utils import Tee

from expl_perf_drop.data import select as data_select
from expl_perf_drop.metrics import select as metric_select
from expl_perf_drop.models import select as model_select


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help = 'Experiment name for downstream tracking purposes')
parser.add_argument('--output_dir', type=str, required = True)
parser.add_argument('--model_dir', type=str, required = True, help = 'only used to get source data hparams')
parser.add_argument('--dataset', type = str, choices = ['synthetic'], required = True)
parser.add_argument('--data_path', type = str, default = '/data/healthy-ml/scratch/haoran/results/expl_perf_drop/data')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_seed', type=int, default=42)
parser.add_argument('--test_pct', type=float, default = 0.25)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--weight_model', type = str, choices = ['xgb', 'lr'], default = 'xgb')
parser.add_argument('--calibrate_weight_models', action = 'store_true')

# only for spurious synthetic (target dataset)
parser.add_argument('--spu_q', type=float, default = 0.5)
parser.add_argument('--spu_n', type=int, default = 20000)
parser.add_argument('--spu_y_noise', type=float, default = 0.25)
parser.add_argument('--spu_mu_add', type=float, default = 3.)
parser.add_argument('--spu_x1_weight', type=float, default = 1.)

# comment out L264-265 of fcm.py in dowhy: if not is_categorical(Y):...

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
# model = pickle.load((model_dir/'model.pkl').open('rb'))

hparams['model_train_hparams'] = model_train_hparams
assert model_train_hparams['dataset'] == args.dataset

with open(os.path.join(args.output_dir,'args.json'), 'w') as f:
    json.dump(hparams, f, indent = 4)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data = data_select(model_train_hparams, device)
source_train_df, source_eval_df = data.get_source_train_test()
target_train_df, target_eval_df = data.get_target_train_test(hparams)

source_df = pd.concat((source_train_df, source_eval_df), ignore_index = True)
target_df = pd.concat((target_train_df, target_eval_df), ignore_index = True)

# source_df['Y'] = source_df['Y'].astype(str)
# target_df['Y'] = target_df['Y'].astype(str)

cont_model = SklearnRegressionModel(model_select(args.weight_model, 'regression'))
dis_model = SklearnClassificationModel(model_select(args.weight_model, 'classification'))

scm = gcm.ProbabilisticCausalModel(data.GRAPH)

scm.set_causal_mechanism('G', gcm.EmpiricalDistribution())
scm.set_causal_mechanism('Y', gcm.ClassifierFCM(classifier_model = dis_model.clone()))
scm.set_causal_mechanism('X1', gcm.AdditiveNoiseModel(prediction_model = cont_model.clone()))
scm.set_causal_mechanism('X2', gcm.AdditiveNoiseModel(prediction_model = cont_model.clone()))
scm.set_causal_mechanism('X3', gcm.AdditiveNoiseModel(prediction_model = cont_model.clone()))

gcm.fit(scm, source_df)

res = estimate_distribution_change_scores(
    scm,
    source_df,
    target_df
)

pickle.dump(
    {
        'expl': res
    },open(os.path.join(args.output_dir,'stats.pkl'), 'wb'))

with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    f.write('done')