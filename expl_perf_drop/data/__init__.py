from expl_perf_drop.data import synthetic, celebA, eicu, cmnist, camelyon

def select(hparams, device):
    if hparams['dataset'] == 'synthetic':
        return synthetic.BackdoorSpurious(hparams)
    elif hparams['dataset'] == 'celebA':
        return celebA.CelebA(hparams, device)
    elif hparams['dataset'] == 'eicu':
        return eicu.eICU(hparams)
    elif hparams['dataset'] == 'cmnist':
        return cmnist.CMNIST(hparams)
    elif hparams['dataset'] == 'camelyon':
        return camelyon.Camelyon17(hparams, device)
    else:
        raise NotImplementedError(hparams['dataset'])