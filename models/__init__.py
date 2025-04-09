from .EENAIC import EENAIC
from mplug import TeacherMPLUG

__factory = {
    'DNAIC': EENAIC,
    'transformer_ar': TeacherMPLUG,
    'transformer_nar': EENAIC,
    'mplug': TeacherMPLUG,
    'ar': TeacherMPLUG,
    'nar': EENAIC,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)