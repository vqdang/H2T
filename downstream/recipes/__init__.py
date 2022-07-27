

import models.mil as MIL

def get_recipes(model_code='clam', **kwargs):
    if model_code == 'transformer-1':
        return MIL.transformer.Transformer(**kwargs)
    elif model_code == 'transformer-2':
        return MIL.transformer.Transformer(**kwargs)
    elif model_code == 'clam':
        return MIL.clam.CLAM_SB(**kwargs)
    elif model_code == 'linear-probe':
        return MIL.clam.CLAM_SB(**kwargs)
    elif model_code == 'cnn-probe':
        from models.probe import Probe
        return Probe(**kwargs)
    else:
        assert False
