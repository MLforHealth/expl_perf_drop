import pandas as pd
import numpy as np
from sklearn.base import clone
from expl_perf_drop.utils import powerset, flatten, check_significant_mechanism_change, map_node_to_feat_names
import shap

class SHAPExplainer():
    def __init__(self, cgm, source_train_df, source_eval_df, target_train_df, target_eval_df,
            subset_features, feature_mapping, target_name = 'Y', background_n_samples = 500, task_type = 'classification'):

        self.source_train_df = source_train_df
        self.target_train_df = target_train_df
        self.source_eval_df = source_eval_df
        self.target_eval_df = target_eval_df
        self.dag = cgm
        self.subset_features = subset_features # ordered list of features to be passed into the model
        self.target_name = target_name
        self.nodes = tuple(self.dag.nodes)
        self.edges = tuple(self.dag.edges)
        self.feature_mapping = feature_mapping
        self.all_columns = flatten(list(self.feature_mapping.values()))
        for i in self.all_columns:
            assert i in source_train_df
            assert i in target_train_df
            assert i in source_eval_df
            assert i in target_eval_df

        for i in self.feature_mapping:
            assert i in self.nodes

        self.background_n_samples = background_n_samples
        self.task_type = task_type
        print("Checking significant mechanism changes...")
        self.p_values = check_significant_mechanism_change(self.dag, feature_mapping, source_train_df, target_train_df)
        self.significant_changes = self.p_values[(self.p_values <= 0.05)].index
        self.significant_features = map_node_to_feat_names(self.significant_changes, self.feature_mapping)
        self.idx_to_take = {i:c for c,i in enumerate(self.subset_features) if i in self.significant_features}

    def explain(self, model, metric, normalize = True, pretty_format = True):
        if self.task_type == 'classification':
            f = lambda x: model.predict_proba(x)[:,1]
        else:
            f = lambda x: model.predict(x)
        explainer = shap.KernelExplainer(f, shap.sample(self.target_train_df[self.subset_features], self.background_n_samples))
        # source_vals = explainer.shap_values(self.source_eval_df)
        target_vals = explainer.shap_values(self.target_eval_df[self.subset_features])
        summarized_vals = np.abs(target_vals).mean(axis = 0)

        if normalize:
            source_perf = metric(model, self.source_eval_df, self.subset_features, target_name = self.target_name)
            target_perf = metric(model, self.target_eval_df, self.subset_features, target_name = self.target_name)
            summarized_vals *= (target_perf - source_perf)/summarized_vals.sum()
        
        feat_val_mapping = pd.Series({i: summarized_vals[c] for i, c in self.idx_to_take.items()})
        
        expl = {}
        for g in self.significant_changes:
            accum = 0
            for feat in self.feature_mapping[g]:
                if feat in feat_val_mapping:
                    accum += feat_val_mapping[feat] 
            
            if pretty_format:
                expl[f'P({g})'] = accum
            else:
                expl[g] = accum

        expl = pd.Series(expl)
        if normalize:
            expl = expl*((target_perf - source_perf)/expl.values.sum())

        return expl

    def get_perf_on_sets(self, model, metric):
        return {
            'source': metric(model, self.source_eval_df, self.subset_features, target_name = self.target_name),
            'target': metric(model, self.target_eval_df, self.subset_features, target_name = self.target_name)
        }