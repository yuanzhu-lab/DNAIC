import importlib
from losses.cross_entropy import CrossEntropy
from losses.distillation_loss import DistillationLoss

__factory = {
    'CrossEntropy': CrossEntropy,
    'DistillationLoss': DistillationLoss,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)