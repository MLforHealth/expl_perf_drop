import numpy as np
from expl_perf_drop.utils import flatten
from expl_perf_drop.utils import Graph
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import timm
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, paths, data_dir, transform):
        super().__init__()
        self.paths = paths
        self.transform = transform
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        fname = self.paths[idx]
        img = Image.open(self.data_dir/'images'/fname)
        return self.transform(img)


class CelebA():
    TARGET_NAME = 'Male'
    TASK_TYPE = 'classification'
    GRAPH = Graph(
        nodes= ['Young', 'Male', 'Eyeglasses', 'Bald', 'Mustache', 'Smiling', 'Wearing_Lipstick', 
            'Mouth_Slightly_Open', 'Narrow_Eyes'],
        edges=[
            ('Male', 'Wearing_Lipstick'),
            ('Male', 'Smiling'),
            ('Male', 'Mustache'),
            ('Male', 'Bald'),
            ('Male', 'Narrow_Eyes'),
            ('Smiling', 'Narrow_Eyes'),
            ('Smiling', 'Mouth_Slightly_Open'),
            ('Young', 'Bald'),
            ('Young', 'Eyeglasses'),
            ('Young', 'Mustache'),
            ('Young', 'Mouth_Slightly_Open'),
            ('Young', 'Narrow_Eyes'),
            ('Young', 'Smiling')
        ]
    )
    VAR_CATEGORIES = {
        i: [i] for i in GRAPH.nodes
    }

    def __init__(self, hparams, device):
        self.hparams = hparams 
        self.device = device
        self.data_seed = hparams['data_seed']
        self.test_pct = self.hparams['test_pct']
        if 'emb_model' in hparams:
            self.cnn_model = hparams['emb_model']
        else:
            self.cnn_model = 'resnet18'        

        if self.cnn_model == 'resnet50':
            resnet_n_features = 2048
        elif self.cnn_model == 'resnet18':
            resnet_n_features = 512

        self.TRAIN_FEATURES = [f'X{i}' for i in range(resnet_n_features)]
        self.train_df, self.test_df = self.load_or_calc_embs(Path(hparams['data_dir'])/'celebA'/'base', hparams['data_seed'], hparams['test_pct'])
        
    def load_or_calc_embs(self, data_dir, seed, test_pct, cache_embs = True):
        if (data_dir/f'train_df_{self.cnn_model}.pkl').is_file():
            train_df = pd.read_pickle(data_dir/f'train_df_{self.cnn_model}.pkl')
            test_df = pd.read_pickle(data_dir/f'test_df_{self.cnn_model}.pkl')
        else:
            m = timm.create_model(self.cnn_model, pretrained=True, num_classes=0).to(self.device).eval()
            df = pd.read_csv(data_dir/'labels.csv')
            paths = df['file_path'].values
            transform=transforms.Compose(
                    [transforms.Resize((224, 224)), transforms.ToTensor()]
                )

            dataset = ImgDataset(paths, data_dir, transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = 256, shuffle = False)
            embs = self.calc_embs(m, dataloader)

            for i in range(embs.shape[-1]):
                df[f'X{i}'] = embs[:, i]

            train_df, test_df = train_test_split(df, test_size = test_pct, random_state = seed)

            if cache_embs:
                train_df.to_pickle(data_dir/f'train_df_{self.cnn_model}.pkl')
                test_df.to_pickle(data_dir/f'test_df_{self.cnn_model}.pkl')

        return train_df, test_df

    def calc_embs(self, m, loader):
        embs = []
        with torch.no_grad():
            for x in tqdm(loader):
                x = x.to(self.device)
                embs.append(m(x).detach().cpu().numpy())
        return np.concatenate(embs)

    def get_source_train_test(self):
        return self.train_df, self.test_df

    def get_target_train_test(self, shift_hparams):
        assert shift_hparams['target_data_dir'] != self.hparams['data_dir']
        train_df, test_df = self.load_or_calc_embs(Path(shift_hparams['target_data_dir']), shift_hparams['data_seed'], shift_hparams['test_pct'])
        return train_df, test_df
