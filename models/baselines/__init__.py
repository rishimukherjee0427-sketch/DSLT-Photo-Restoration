from models.baselines.restormer import (
    Restormer, build_restormer, pad_to_multiple, restore_full, set_grad_checkpoint)
from models.baselines.promptir import PromptIR, build_promptir

BUILDERS = {'restormer': build_restormer, 'promptir': build_promptir}


def build_baseline(name, size='small', **kw):
    return BUILDERS[name](size, **kw)
