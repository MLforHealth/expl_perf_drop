import os
import numpy as np
from expl_perf_drop.utils import flatten
from expl_perf_drop.utils import Graph
import pandas as pd
from sklearn.model_selection import train_test_split

class eICU():
    TARGET_NAME = 'death'
    VAR_CATEGORIES = {
        'Demo': [
            'is_female',
            'race_black', 
            'race_hispanic', 
            'race_asian', 
            'race_other',
        ],  # demographics
        'Vitals': [
            'heartrate',
            'sysbp',
            'temp',
            'bg_pao2fio2ratio',
            'urineoutput',
        ],  # vitals
        'Labs': [
            'bun',
            'sodium',
            'potassium',
            'bicarbonate',
            'bilirubin',
            'wbc',
            'gcs',
        ],  # labs
        'Age': [
            'age',
        ],  # miscellaneous
        'ElectiveSurgery': [
            'electivesurgery',
        ],  # miscellaneous
        'Outcome': ['death']
    }
    TASK_TYPE = 'classification'
    GRAPH = Graph(
        nodes= list(VAR_CATEGORIES.keys()),
        edges=[
            ('Demo', 'Outcome'),
            ('Vitals', 'Outcome'), 
            ('Labs', 'Outcome'),
            ('Age', 'Outcome'), 
            ('ElectiveSurgery', 'Outcome'), 
            ('Demo', 'Vitals'),
            ('Age', 'Vitals'), 
            ('Vitals', 'ElectiveSurgery'),
            ('Demo', 'Labs'),
            ('Age', 'Labs'), 
            ('Labs', 'ElectiveSurgery'),
            ('Age', 'ElectiveSurgery'),
            ('Demo', 'ElectiveSurgery'),
        ]
    )
    def __init__(self, hparams):
        self.hparams = hparams 
        self.data_seed = self.hparams['data_seed']
        self.n = self.hparams['n']
        self.test_pct = self.hparams['test_pct']
        self.source_hospital_ids = self.hparams['source_hospital_ids']
        self.target_hospital_ids = self.hparams['target_hospital_ids']
        rng = np.random.RandomState(self.data_seed)
        self.w = rng.normal(size = (3, 1))
        self.TRAIN_FEATURES = [j for i in self.VAR_CATEGORIES for j in self.VAR_CATEGORIES[i] if i!='Outcome']

    def generate(self, n, rng, hospital_ids):
        df = pd.read_csv(os.path.join(self.hparams['data_dir'],'data_eicu_extract.csv'))
        df = df[df['hospitalid'].isin(hospital_ids)]
        df = df[:n]
        cols = self.VAR_CATEGORIES['Demo'] + self.VAR_CATEGORIES['Vitals'] + self.VAR_CATEGORIES['Labs'] + self.VAR_CATEGORIES['Age'] + self.VAR_CATEGORIES['ElectiveSurgery'] + self.VAR_CATEGORIES['Outcome']
        df = df[cols]
        return df

    def get_source_train_test(self):
        rng = np.random.RandomState(self.data_seed)
        df = self.generate(self.n, rng, hospital_ids=self.source_hospital_ids)
        return train_test_split(df, random_state = self.data_seed, shuffle = True, test_size = self.test_pct)
    
    def get_subsampled_source_train_test(self, q=0.5):
        rng = np.random.RandomState(self.data_seed)
        df = self.generate(self.n, rng, hospital_ids=self.source_hospital_ids)
        df.loc[:, "weights"] = 1
        qquantile = np.quantile(df["age"], q=q)
        print(f"{q}-quantile age {qquantile}")
        df.loc[df["age"]<qquantile, "weights"] = 5
        df = df.sample(n=int(df.shape[0]*0.67), weights=df["weights"], random_state = self.data_seed)
        df = df.drop(labels=["weights"], axis=1)
        return train_test_split(df, random_state = self.data_seed, shuffle = True, test_size = self.test_pct)

    def get_target_train_test(self, shift_hparams):
        assert shift_hparams['data_seed'] != self.hparams['data_seed']
        rng = np.random.RandomState(shift_hparams['data_seed'])
        df = self.generate(shift_hparams['spu_n'], rng, hospital_ids=self.target_hospital_ids)
        return train_test_split(df, random_state = shift_hparams['data_seed'], shuffle = True, test_size = shift_hparams['test_pct'])
