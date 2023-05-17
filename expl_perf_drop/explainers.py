from sklearn.linear_model import LogisticRegression, LinearRegression
from itertools import combinations
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, mean_squared_error
from sklearn.calibration import calibration_curve
import scipy.stats
import pandas as pd
import numpy as np
from sklearn.base import clone
import copy
from expl_perf_drop.utils import powerset, flatten, df_to_domain, str_to_dist
from expl_perf_drop.models import select as model_select
from sklearn.calibration import CalibratedClassifierCV
from expl_perf_drop.shapley import estimate_shapley_values, ShapleyApproximationMethods, ShapleyConfig
from expl_perf_drop.KLIEP import DensityRatioEstimator

factorial = np.math.factorial

class CGExplainer():
    def __init__(self, cgm, source_train_df, source_eval_df, target_train_df, target_eval_df,
            subset_features, feature_mapping, target_name = 'Y', model_type_dis = LogisticRegression(random_state = 42), 
                 model_type_cont = LinearRegression(), shapley_method = 'AUTO', exclude_dists = []):
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
        self.exclude_dists = []
        for i in exclude_dists:
            if isinstance(i, str):
                self.exclude_dists.append(str_to_dist(i))
            else:
                assert ValueError("Pass in excluded distribution as a string! (e.g. 'P(Y|X)')")
        if shapley_method == 'debug':
            self.shapley_config = None
        else:
            self.shapley_config = ShapleyConfig(approximation_method = getattr(ShapleyApproximationMethods, shapley_method))
        # assert set(self.subset_features).issubset(set(self.all_columns))
        assert set(self.subset_features).issubset(set(source_train_df.columns))
        
        for i in self.all_columns:
            assert i in source_train_df
            assert i in target_train_df
            assert i in source_eval_df
            assert i in target_eval_df

        for i in self.feature_mapping:
            assert i in self.nodes
                    
        self.root_nodes = [i for i in self.dag.nodes if self.dag.in_degree(i) == 0]
        self.feature_types = {}
        for i in self.all_columns:
            if np.array_equal(np.unique(source_train_df[i]), np.array([0, 1])):
                self.feature_types[i] = 'binary'
            elif len(np.unique(source_train_df[i]))<= 20:
                self.feature_types[i] = 'categorical'
            else:
                self.feature_types[i] = 'continuous'
        
        self.model_type_cont = model_type_cont
        self.model_type_dis = model_type_dis
        
    def get_incoming_nodes(self, i):
        return tuple([j[0] for j in self.dag.in_edges(i)])
    
    def pretty_format_shift(self, i):
        if isinstance(i, tuple): # conditional
            return f'P({i[1]} | {",".join(sorted(i[0]))})'
        else: # marginal
            return f'P({i})'

    def get_perf_on_sets(self, model, metric):
        return {
            'source': metric(model, self.source_eval_df, self.subset_features, target_name = self.target_name),
            'target': metric(model, self.target_eval_df, self.subset_features, target_name = self.target_name)
        }
    
    def get_all_possible_shifts(self):
        shifts = copy.copy(self.root_nodes) # marginal shifts
        for i in self.nodes:
            if i not in self.root_nodes: # conditional shifts
                in_nodes = self.get_incoming_nodes(i)
                key = (frozenset(in_nodes), i) # TODO: represent this as an object instead
                shifts.append(key)
        return [i for i in shifts if i not in self.exclude_dists]

    def scale(self, expl, perfs):
        delta = perfs['target'] - perfs['source']
        return expl*(delta/expl.values.sum())
        
class CGExplainerDR(CGExplainer):
    def __init__(self, cgm, source_train_df, source_eval_df, target_train_df, target_eval_df,
            subset_features, feature_mapping, target_name = 'Y', model_type_dis = model_select('xgb', 'classification'), 
                 model_type_cont = LinearRegression(), density_estimator = 'proba', calibrate_weight_models = False,
                 clip_weight_thres = None, clip_prob_thres = None, imp_weight_type = 'normal', shapley_method = 'AUTO',
                 exclude_dists = []):
        f'''
        cgm: expl_perf_drop.utils.Graph
            Causal graphical model.
        source_train_df : pandas.DataFrame
            Training set on source environment, used to train the importance weighting models
        target_train_df : pandas.DataFrame
            Training data on target environment, used to train the importance weighting models
        source_eval_df : pandas.DataFrame
            Source evaluation data.
        target_eval_df : pandas.DataFrame
            Target evaluation data.
        subset_features : list
            List of features to be passed into the model. Each item must be a column in all dataframes.
        feature_mapping : dict
            A dictionary mapping from node names to feature names.
        model_type_dis : sklearn model object
            Model type for importance weighting.
        model_type_cont : deprecated, can ignore
        density_estimator : str
            Type of importance weight estimator. Can be "proba" or "kliep".
        calibrate_weight_models : bool
            Whether to post-hoc calibrate the weight models.
        clip_weight_thres : float
            Clip weights to be within [1/clip_weight_thres, clip_weight_thres]. Should be >= 1.
        clip_prob_thres : float
            Clip probabilties (from importance estimators) to be [1-clip_prob_thres, clip_prob_thres]. Should be >= 0.5.
        imp_weight_type : str
            The type of importance weighting to be used. Can be either 'normal' or 'self_normalize'.
        shapley_method : str
            Method for calculating Shapley values. Can be "AUTO", "debug", "EXACT", "EXACT_FAST", "SUBSET_SAMPLING", "EARLY_STOPPING", or "PERMUTATION".
        exclude_dists : list
            List of distributions to exclude (i.e. do not take these into account).
        '''
        super().__init__(cgm, source_train_df, source_eval_df, target_train_df, target_eval_df,
            subset_features, feature_mapping, target_name, model_type_dis, model_type_cont, shapley_method, exclude_dists)
        self.density_estimator = density_estimator  
        self.weight_models_trained = False
        self.calibrate_weight_models = calibrate_weight_models
        self.clip_weight_thres = clip_weight_thres
        self.clip_prob_thres = clip_prob_thres
        if clip_weight_thres is not None:
            assert clip_weight_thres >= 1
        if clip_prob_thres is not None:
            assert clip_prob_thres >= 0.5
        self.imp_weight_type = imp_weight_type
    
    def explain(self, model, metric, pretty_format = True):
        '''
        model : The model to be explained.
        metric : callable
            The metric to be used for explanation.
            Must take in (model, dataframe, subset_features, weight, target_name) and return a scalar.
        '''
        self.cache = {'weights': {}, 'deltas': {}, 'pct_weight_clip': {}}
        if not self.weight_models_trained:
            self._train_weight_models()
        shifts = self.get_all_possible_shifts()
        self.cache['all_shifts'] = shifts

        source_metric = metric(model, self.source_eval_df, self.subset_features, target_name = self.target_name)

        if self.shapley_config is None:
            values = {}
            for i in shifts:
                val = 0
                V_minus_i = [j for j in shifts if j != i]
                for S in powerset(V_minus_i):
                    val += ((self._delta(S + [i], model, metric, source_metric) - self._delta(S, model, metric, source_metric)) 
                        * (factorial(len(S)) * factorial(len(shifts) - len(S) - 1))/factorial(len(shifts)))
                values[self.pretty_format_shift(i) if pretty_format else i] = val
        else:
            values_list = estimate_shapley_values(lambda x: self._delta([shifts[i] for i in x.nonzero()[0]], model, metric, source_metric),
                num_players = len(shifts),
                shapley_config = self.shapley_config)
            values = {self.pretty_format_shift(i) if pretty_format else i : j for i, j in zip(shifts, values_list)}

        return pd.Series(values)
        
    def _density_ratio_proba(self, source, target):
        X, Y = df_to_domain(source, target)
        clf = clone(self.model_type_dis).fit(X, Y)
        best_score = clf.best_score_

        if self.calibrate_weight_models:
            clf = CalibratedClassifierCV(base_estimator = clf, method = 'isotonic', cv = 'prefit').fit(X, Y)
        
        def ratio_model(x):
            prob = clf.predict_proba(x)
            if self.clip_prob_thres is not None:
                prob = np.clip(prob, 1-self.clip_prob_thres, self.clip_prob_thres)

            ratio = prob[:,1]/prob[:,0]
            return ratio

        pred_proba = clf.predict_proba(X)[:, 1]
        return ratio_model, {'roc': roc_auc_score(Y, pred_proba),
                            'roc_grid': best_score,
                            'brier': brier_score_loss(Y, pred_proba),
                            'clf': clf}
    
    def _density_ratio_kliep(self, source, target):
        kliep = DensityRatioEstimator()
        kliep.fit(source, target) 
        def ratio_model(x):
            ratio = kliep.predict(x)
            return ratio
        return ratio_model, {}

    def _map_node_to_feat_names(self, nodes):
        return sorted([j for i in nodes for j in self.feature_mapping[i]])

    def _train_weight_model(self, density_ratio_fn, nodes):
        return density_ratio_fn(self.source_train_df[self._map_node_to_feat_names(list(nodes))].values,
                                         self.target_train_df[self._map_node_to_feat_names(list(nodes))].values)

    def _train_weight_models(self, extra_models = []):
        if self.density_estimator == 'proba':
            density_ratio_fn = self._density_ratio_proba
        elif self.density_estimator == 'kliep':
            density_ratio_fn = self._density_ratio_kliep
        else:
            raise NotImplementedError  
        self.weight_models = {}
        self.weight_model_accs = {}
        for i in self.nodes:
            if i in self.root_nodes:
                if i in self.exclude_dists:
                    continue
                self.weight_models[frozenset([i])], self.weight_model_accs[frozenset([i])] = self._train_weight_model(density_ratio_fn, [i])
            else:      
                in_nodes = self.get_incoming_nodes(i)
                if (frozenset(in_nodes), i) in self.exclude_dists:
                    continue
                if frozenset(in_nodes) not in self.weight_models:
                    self.weight_models[frozenset(in_nodes)], self.weight_model_accs[frozenset(in_nodes)] = self._train_weight_model(density_ratio_fn, list(in_nodes))
                if frozenset((*in_nodes,i)) not in self.weight_models:
                    self.weight_models[frozenset((*in_nodes,i))], self.weight_model_accs[frozenset((*in_nodes,i))] = self._train_weight_model(density_ratio_fn, list(in_nodes) + [i])
        self.weight_models_trained = True

        for i in extra_models:
            nodes = [i] if isinstance(i, str) else i
            self.weight_models[frozenset(nodes)], self.weight_model_accs[frozenset(nodes)] = self._train_weight_model(density_ratio_fn, nodes)

        return self.weight_models, self.weight_model_accs

    def _evaluate_weight_models_on_eval(self):
        res = {}
        for i in self.weight_models:
            cols = self._map_node_to_feat_names(list(i))
            X, Y = df_to_domain(self.source_eval_df[cols], self.target_eval_df[cols])
            preds = self.weight_model_accs[i]['clf'].predict_proba(X)[:, 1]
            res[i] = {
                'preds': preds,
                'Y': Y,
                'roc': roc_auc_score(Y, preds),
                'brier': brier_score_loss(Y, preds),
            }
        return res
        
    def _delta(self, S, model, metric, source_metric = None, return_weights = False, use_cache = True):
        if use_cache and frozenset(S) in self.cache['deltas']:
            return self.cache['deltas'][frozenset(S)]

        n_source = self.source_eval_df.shape[0]
        n_target = self.target_eval_df.shape[0]
        weight = np.ones((n_source,)) 

        for c, i in enumerate(self.dag.nodes):
            if i in self.root_nodes:
                if i in S:
                    if len(self._map_node_to_feat_names([i]))==1:  # perhaps reshape(-1,1) is not needed since df[['single feature']] returns (n,1) array
                        weight *= self.weight_models[frozenset([i])](self.source_eval_df[self._map_node_to_feat_names([i])].values.reshape(-1,1)) * n_source/n_target
                    else:
                        weight *= self.weight_models[frozenset([i])](self.source_eval_df[self._map_node_to_feat_names([i])].values) * n_source/n_target
            else:
                in_nodes = self.get_incoming_nodes(i)
                key = (frozenset(in_nodes), i)
                if key in S:
                    weight *= (self.weight_models[frozenset((*in_nodes,i))](self.source_eval_df[self._map_node_to_feat_names(list(in_nodes) + [i])].values) / 
                            self.weight_models[frozenset(in_nodes)](self.source_eval_df[self._map_node_to_feat_names(list(in_nodes))].values))

        if self.clip_weight_thres is not None:
            if use_cache:
                self.cache['pct_weight_clip'][frozenset(S)] = ((weight < 1/self.clip_weight_thres) | (weight > self.clip_weight_thres)).sum()/len(weight)
            weight = np.clip(weight, 1/self.clip_weight_thres, self.clip_weight_thres)

        if source_metric is None:
            source_metric = metric(model, self.source_eval_df, self.subset_features, target_name = self.target_name)

        target_metric = metric(model, self.source_eval_df, self.subset_features, weight, target_name = self.target_name)
        delta = target_metric - source_metric
        
        if self.imp_weight_type == 'normal':
            pass
        elif self.imp_weight_type == 'self_normalize':
            delta = delta * len(weight) / sum(weight)
        else:
            raise NotImplementedError(self.imp_weight_type)
            
        if use_cache:
            self.cache['weights'][frozenset(S)] = weight
            self.cache['deltas'][frozenset(S)] = delta

        if return_weights:
            return delta, weight
        return delta