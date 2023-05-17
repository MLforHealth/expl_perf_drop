import argparse
import collections
import pickle
import json
import os
import random
import sys
import time
import uuid
import socket

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pathlib import Path
from expl_perf_drop.utils import Tee, eval_metrics
from expl_perf_drop.data import select as data_select
from expl_perf_drop.models import select as model_select


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, help = 'Experiment name for downstream tracking purposes')
parser.add_argument('--output_dir', type=str, required = True)
parser.add_argument('--dataset', type = str, choices = ['synthetic', 'celebA', 'cmnist', 'camelyon'], required = True)
parser.add_argument('--emb_model', type = str, choices = ['resnet50', 'resnet18'], default = 'resnet18')
parser.add_argument('--data_dir', type = str, default = '/home/gridsan/hrzhang/results/expl_perf_drop/data')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--test_pct', type=float, default = 0.25)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--model', type = str, choices = ['xgb', 'lr', 'nn', 'jtt_nn', 'gdro_nn'], default = 'xgb')

# spurious synthetic (source dataset)
parser.add_argument('--n', type=int, default = 20000)
parser.add_argument('--spu_q', type=float, default = 0.9)
parser.add_argument('--spu_y_noise', type=float, default = 0.25)
parser.add_argument('--spu_mu_add', type=float, default = 3.)
parser.add_argument('--spu_x1_weight', type=float, default = 1.)

# Camelyon-17
parser.add_argument('--camelyon_source_site', type=int, default = 0, choices = [0, 1, 2, 3, 4])

args = parser.parse_args()
hparams = vars(args)

os.makedirs(args.output_dir, exist_ok=True)
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

with open(os.path.join(args.output_dir,'args.json'), 'w') as f:
    json.dump(hparams, f, indent = 4)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data = data_select(hparams, device)
train_df, test_df = data.get_source_train_test()
if args.model == 'gdro_nn':
    model = model_select(hparams['model'], data.TASK_TYPE, hparams, device).fit(train_df[data.TRAIN_FEATURES], train_df[data.TARGET_NAME], train_df['A'])
else:
    model = model_select(hparams['model'], data.TASK_TYPE, hparams, device).fit(train_df[data.TRAIN_FEATURES], train_df[data.TARGET_NAME])
pickle.dump(model, open(os.path.join(args.output_dir,'model.pkl'), 'wb'))
pickle.dump(eval_metrics(test_df[data.TARGET_NAME], model, data.TASK_TYPE, data.TRAIN_FEATURES, test_df), 
        open(os.path.join(args.output_dir,'stats.pkl'), 'wb'))

with open(os.path.join(args.output_dir, 'done'), 'w') as f:
    f.write('done')
