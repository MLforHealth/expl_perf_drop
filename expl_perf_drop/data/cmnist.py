import numpy as np
from expl_perf_drop.utils import flatten
from expl_perf_drop.utils import Graph
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchvision import datasets
from tqdm import tqdm
import torch

class CMNIST():
    TARGET_NAME = 'Y'
    VAR_CATEGORIES = {
        'A': ['A'],
        'Y': ['Y'],
        'X': [f'X{i}' for i in range(2*14*14)]
    }
    TRAIN_FEATURES = VAR_CATEGORIES['X']  
    TASK_TYPE = 'classification'
    GRAPH = Graph(
        nodes= list(VAR_CATEGORIES.keys()),
        edges=[
            ('Y', 'A'),
            ('A', 'X'),
            ('Y', 'X')
        ]
    )

    def __init__(self, hparams):
        self.hparams = hparams 
        self.data_seed = hparams['data_seed']

        root = Path(hparams['data_dir'])/'cmnist'
        mnist = datasets.MNIST(root, train=True)       
        self.X_src, self.y_src = mnist.data[:20000], mnist.targets[:20000]
        self.X_tar, self.y_tar = mnist.data[20000:40000], mnist.targets[20000:40000]
        self.test_pct = hparams['test_pct']

    def subsample(self, mask, n_samples, rng, imgs, color, binary_label):
        assert n_samples <= mask.sum()
        idxs = np.concatenate((np.nonzero(~mask)[0], rng.choice(np.nonzero(mask)[0], size=n_samples, replace=False)))
        rng.shuffle(idxs)
        return imgs[idxs], color[idxs], binary_label[idxs]        

    def generate(self, rng, X, y, label_prob, color_prob, flip_prob):
        binary_label = np.bitwise_xor(y >= 5, (rng.random(len(y)) < flip_prob)).numpy()
        color = np.bitwise_xor(binary_label, (rng.random(len(y)) < color_prob))
        imgs = torch.stack([X, X], dim=1).numpy()
        imgs[list(range(len(imgs))), (1 - color), :, :] *= 0

        # subsample color = 0
        if label_prob > 0.5:
            n_samples_0 = int((binary_label == 1).sum() * (1-label_prob) /label_prob)
            imgs, color, binary_label = self.subsample(binary_label == 0, n_samples_0, rng, imgs, color, binary_label)
        # subsample color = 1
        elif label_prob < 0.5:
            n_samples_1 = int((binary_label == 0).sum() * label_prob / (1-label_prob))
            imgs, color, binary_label = self.subsample(binary_label == 1, n_samples_1, rng, imgs, color, binary_label)

        imgs = imgs[:, :, ::2, ::2] # 2x subsample for computational convenience
        imgs = (imgs/255).astype(float)
        imgs = imgs.reshape(len(imgs), -1)
        return self.data_to_df(imgs, color, binary_label)

    def data_to_df(self, X, A, Y):
        df = pd.DataFrame({'A': A.astype(int), 'Y': Y.astype(int)})
        for i in range(0, X.shape[1]):
            df[f'X{i}'] = X[:, i]
        return df         

    def get_source_train_test(self):
        rng = np.random.RandomState(self.data_seed)
        df = self.generate(rng, self.X_src, self.y_src, 0.50, 0.15, 0.25)
        return train_test_split(df, random_state = self.data_seed, shuffle = True, test_size = self.test_pct)

    def get_target_train_test(self, shift_hparams):
        rng = np.random.RandomState(shift_hparams['data_seed'])
        df = self.generate(rng, self.X_tar, self.y_tar, shift_hparams['cmnist_label_prob'], shift_hparams['cmnist_color_prob'], 
             shift_hparams['cmnist_flip_prob'])
        return train_test_split(df, random_state = shift_hparams['data_seed'], shuffle = True, test_size = shift_hparams['test_pct'])