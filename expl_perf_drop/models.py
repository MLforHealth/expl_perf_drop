
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch
from expl_perf_drop.data import select as data_select

class TorchModel(nn.Module):
    def __init__(self, model_hparams, device):
        super().__init__()
        self.lr = model_hparams['lr']
        self.batch_size = model_hparams['batch_size']
        self.n_epochs = model_hparams['n_epochs']
        self.device = device 
        self.debug = model_hparams['debug']

    def fit(self, X, y, weights = None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if weights is None:
            train_ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1), torch.ones((len(y), 1)))
        else:
            train_ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1), weights)
        train_loader = DataLoader(train_ds, batch_size = self.batch_size, shuffle = True)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )     

        for epoch in range(self.n_epochs):
            self.train()
            train_loss = []
            for x, lab, weights in train_loader:
                x = x.float().to(self.device)
                lab = lab.float().to(self.device)
                pred = self(x)
                loss = F.binary_cross_entropy_with_logits(pred, lab, weight = weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            if self.debug:
                print(f'Epoch {epoch}; Train loss: {np.mean(train_loss)}')

        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.float().to(self.device)
        self.eval()

        with torch.no_grad():
            out = torch.sigmoid(self(X)).squeeze(1).detach().cpu().numpy()
        return np.stack((1 - out, out), axis = 1)

    def predict(self, X):
        self.eval()
        return self.predict_proba(X)[:, 1].round()

class MLP(TorchModel):
    def __init__(self, n_inputs, n_outputs, model_hparams, device):
        super(MLP, self).__init__(model_hparams, device)
        self.input = nn.Linear(n_inputs, model_hparams['mlp_width'])
        self.dropout = nn.Dropout(model_hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(model_hparams['mlp_width'],model_hparams['mlp_width'])
            for _ in range(model_hparams['mlp_depth']-2)])
        self.output = nn.Linear(model_hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class JTT_MLP():
    def __init__(self, n_inputs, n_outputs, model_hparams, device):
        self.device = device
        self.jtt_lambda = model_hparams['jtt_lambda']
        self.erm_model = MLP(n_inputs, n_outputs, model_hparams, device)
        self.robust_model = MLP(n_inputs, n_outputs, model_hparams, device)

    def fit(self, X, y):
        if isinstance(y, pd.Series):
            y = y.values
        self.erm_model.fit(X, y)
        self.erm_model.eval()
        predictions = self.erm_model.predict(X)        
        wrong_predictions = predictions != y
        weights = torch.ones(wrong_predictions.shape).to(self.device).float()
        weights[wrong_predictions == 1] = self.jtt_lambda
        self.robust_model.fit(X, y, weights.unsqueeze(-1))
        return self

    def predict_proba(self, X):
        self.robust_model.eval()
        return self.robust_model.predict_proba(X)

    def predict(self, X):
        self.robust_model.eval()
        return self.robust_model.predict(X)

class GDRO_MLP(MLP):
    def __init__(self, n_inputs, n_outputs, model_hparams, device):
        super().__init__(n_inputs, n_outputs, model_hparams, device)
        self.eta = model_hparams['groupdro_eta']

    def fit(self, X, y, a):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            a = a.values
        g = np.array([str(c)+str(d) for c,d in zip(y, a)])
        unique_g = np.unique(g)
        g_enc = np.zeros(g.shape)
        for c, i in enumerate(unique_g):
            g_enc[g == i] = c
        unique_g = list(range(len(unique_g)))
        q = torch.ones(len(unique_g)).to(self.device)

        train_ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1), torch.tensor(g_enc).unsqueeze(-1))
        train_loader = DataLoader(train_ds, batch_size = self.batch_size, shuffle = True)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )

        for epoch in range(self.n_epochs):
            self.train()
            train_loss = []
            for x, lab, g_b in train_loader:
                x = x.float().to(self.device)
                lab = lab.float().to(self.device)

                losses = torch.zeros(len(unique_g)).to(self.device)
                errs = torch.zeros(len(unique_g)).to(self.device)
                for c, g_i in enumerate(unique_g):
                    mask = (g_b == g_i).squeeze()
                    x_g, lab_g = x[mask], lab[mask]
                    pred = self(x_g)
                    losses[c] = F.binary_cross_entropy_with_logits(pred, lab_g)
                    errs[c] = ((pred > 0) != lab_g).sum()/mask.sum()
                    # q[c] *= (self.eta * losses[c].data).exp()
                    q[c] *= (self.eta * errs[c].data).exp()
                q /= q.sum()
                loss = torch.dot(losses, q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())                

            if self.debug:                
                print(f'Epoch {epoch}; Train loss: {np.mean(train_loss)}')
                print(losses.detach(), q, errs)

            if max(errs) < 0.3:
                break

        return self


def select(model_type, task = 'classification', hparams = None, device = 'cpu'):
    if model_type == 'xgb' and task == 'classification':
        return GridSearchCV(
            estimator = XGBClassifier(random_state = 42, n_jobs = -1),
            param_grid = {'max_depth': [1, 2, 3, 4, 5]},
            scoring = 'roc_auc_ovr',
            cv = 3,
            refit = True
        )
    elif model_type == 'xgb' and task == 'regression':
        return GridSearchCV(
            estimator = XGBRegressor(random_state = 42, n_jobs = -1, objective ='reg:squarederror'),
            param_grid = {'max_depth': [1, 2, 3, 4, 5]},
            scoring = 'neg_mean_absolute_error',
            cv = 3,
            refit = True
        )
    elif model_type == 'lr' and task == 'classification':
        return GridSearchCV(
            estimator = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(random_state = 42))
            ]),
            param_grid = {'model__C': np.logspace(-4, 1, 20)},
            scoring = 'roc_auc_ovr',
            cv = 3,
            refit = True,
            n_jobs = -1
        )
    elif model_type == 'lr' and task == 'regression':
        return GridSearchCV(
            estimator = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(random_state = 42))
            ]),
            param_grid = {'model__alpha': np.logspace(0, 4, 20)},
            scoring = 'neg_mean_absolute_error',
            cv = 3,
            refit = True,
            n_jobs = -1
        )
    elif model_type == 'nn' and task == 'classification':        
        return MLP(
            len(data_select(hparams, device).TRAIN_FEATURES),
            1,
            {
                'mlp_width': 256,
                'mlp_depth': 3,
                'mlp_dropout': 0.2,
                'lr': 0.001,
                'batch_size': 2048,
                'n_epochs': 100,
                'debug': hparams['debug']
            },
            device
        )
    elif model_type == 'jtt_nn' and task == 'classification':
         return JTT_MLP(
            len(data_select(hparams, device).TRAIN_FEATURES),
            1,
            {
                'mlp_width': 256,
                'mlp_depth': 3,
                'mlp_dropout': 0.2,
                'lr': 0.001,
                'batch_size': 2048,
                'n_epochs': 100,
                'debug': hparams['debug'],
                'jtt_lambda': 100
            },
            device
        )
    elif model_type == 'gdro_nn' and task == 'classification':
        return GDRO_MLP(
            len(data_select(hparams, device).TRAIN_FEATURES),
            1,
            {
                'mlp_width': 256,
                'mlp_depth': 3,
                'mlp_dropout': 0.3,
                'lr': 0.001,
                'batch_size': 2048*2,
                'n_epochs': 30,
                'debug': hparams['debug'],
                'groupdro_eta': 1e-1
            },
            device
        )
    else:
        raise NotImplementedError(model_type)