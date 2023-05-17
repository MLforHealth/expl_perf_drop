import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from expl_perf_drop.utils import Graph
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import timm
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Subset
from pathlib import Path
from wilds.datasets.wilds_dataset import WILDSSubset

class Camelyon17():
    TARGET_NAME = 'Y'    
    TASK_TYPE = 'classification'    

    def __init__(self, hparams, device, cache_embs = True):
        self.device = device
        self.hparams = hparams 
        self.data_seed = hparams['data_seed']
        self.source_site = hparams['camelyon_source_site']
        self.test_pct = hparams['test_pct']
        if 'emb_model' in hparams:
            self.cnn_model = hparams['emb_model']
        else:
            self.cnn_model = 'resnet18'        
        cache_path = Path(hparams['data_dir'])/'cache'/'camelyon'

        if self.cnn_model == 'resnet50':
            resnet_n_features = 2048
        elif self.cnn_model == 'resnet18':
            resnet_n_features = 512

        if hparams['camelyon_graph_direction'] == 'anticausal':
            self.GRAPH = Graph(
                nodes = ['X', 'Y'],
                edges = [
                    ('Y', 'X')]
            )
        elif hparams['camelyon_graph_direction'] == 'causal':
            self.GRAPH = Graph(
                nodes = ['X', 'Y'],
                edges = [
                    ('X', 'Y')]
            )
        self.VAR_CATEGORIES = {
            'X': [f'X{i}' for i in range(resnet_n_features)],
            'Y': 'Y'
        }
        self.TRAIN_FEATURES = [f'X{i}' for i in range(resnet_n_features)]
        self.ds = get_dataset(dataset='camelyon17', root_dir = hparams['data_dir'])
        self.meta_df = self.ds._metadata_df

        if (cache_path/f'embs_{self.cnn_model}.npy').is_file():
            embs = np.load(cache_path/f'embs_{self.cnn_model}.npy')
        else:
            self.m = timm.create_model(self.cnn_model, pretrained=True, num_classes=0).to(self.device).eval()
            transformed_ds = WILDSSubset(self.ds, list(range(len(self.meta_df))),
             transform = transforms.Compose(
                    [transforms.Resize((224, 224)), transforms.ToTensor()]
                ))

            loader = get_eval_loader("standard", transformed_ds, batch_size=256, num_workers = 8)

            embs, y = self.get_embs(loader)
            assert embs.shape[1] == resnet_n_features
            assert (y == self.meta_df['tumor'].values).all()

            if cache_embs:
                cache_path.mkdir(parents = True, exist_ok = True)
                np.save(str(cache_path/f'embs_{self.cnn_model}.npy'), embs)

        self.site_dfs = {}
        for site in [0, 1, 2, 3, 4]:
            mask = self.meta_df['center'] == site
            self.site_dfs[site] = self.data_to_df(embs[mask], self.meta_df['tumor'].values[mask])

    def data_to_df(self, embs, y):
        df = pd.DataFrame({'Y': y})
        for i in range(embs.shape[-1]):
            df[f'X{i}'] = embs[:, i]
        return df

    def get_embs(self, loader):
        embs, y = [], []
        with torch.no_grad():
            for x, y_true, metadata in tqdm(loader):
                x = x.to(self.device)
                y.append(y_true.numpy())
                embs.append(self.m(x).detach().cpu().numpy())
        return np.concatenate(embs), np.concatenate(y)

    def get_source_train_test(self):
        return train_test_split(self.site_dfs[self.source_site].sample(n = 10000, random_state = self.data_seed),
                    random_state = self.data_seed, shuffle = True, test_size = self.test_pct)

    def get_target_train_test(self, shift_hparams):
        return train_test_split(self.site_dfs[shift_hparams['camelyon_target_site']].sample(n = 10000, random_state = shift_hparams['data_seed']),
                    random_state = shift_hparams['data_seed'], shuffle = True, test_size = shift_hparams['test_pct'])
