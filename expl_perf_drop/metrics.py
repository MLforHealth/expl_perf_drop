from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, mean_squared_error, log_loss, recall_score

def accuracy(model, data, subset_cols = None, weight = None, target_name = 'Y'):
    data_input = data[subset_cols] if subset_cols is not None else data
    return accuracy_score(data[target_name],  model.predict(data_input), sample_weight = weight)

def AUROC(model, data, subset_cols = None, weight = None, target_name = 'Y'):
    data_input = data[subset_cols] if subset_cols is not None else data
    return roc_auc_score(data[target_name], model.predict_proba(data_input)[:, 1], sample_weight = weight)

def brier(model, data, subset_cols = None, weight = None, target_name = 'Y'):
    data_input = data[subset_cols] if subset_cols is not None else data
    return brier_score_loss(data[target_name], model.predict_proba(data_input)[:, 1], sample_weight = weight)

def MSE(model, data, subset_cols = None, weight = None, target_name = 'Y'):
    data_input = data[subset_cols] if subset_cols is not None else data
    return mean_squared_error(data[target_name], model.predict(data_input.values), sample_weight = weight)

def TPR(model, data, subset_cols = None, weight = None, target_name = 'Y'):
    data_input = data[subset_cols] if subset_cols is not None else data
    return recall_score(data[target_name], model.predict(data_input.values), sample_weight = weight)

def TNR(model, data, subset_cols = None, weight = None, target_name = 'Y'):
    data_input = data[subset_cols] if subset_cols is not None else data
    return recall_score(data[target_name], model.predict(data_input.values), sample_weight = weight, pos_label = 0)

def BCE(model, data, subset_cols = None, weight = None, target_name = 'Y'):
    data_input = data[subset_cols] if subset_cols is not None else data
    return log_loss(data[target_name], model.predict_proba(data_input.values)[:, 1], sample_weight = weight)

def select(metr):
    '''
    Returns metric function from string.
    Can be "acc", "auroc", "brier", "mse", "bce", "tpr", "tnr".
    With the exception of "mse", assumes that your model is a classification model, and has 
        predict_proba and predict functions with inputs/outputs similar to sklearn.
    '''
    if metr == 'acc':
        return accuracy
    elif metr == 'auroc':
        return AUROC
    elif metr == 'brier':
        return brier
    elif metr == 'mse':
        return MSE
    elif metr == "bce":
        return BCE
    elif metr == 'tpr':
        return TPR
    elif metr == 'tnr':
        return TNR
    else:
        raise NotImplementedError(metr)