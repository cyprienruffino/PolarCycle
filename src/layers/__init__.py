from __future__ import absolute_import
from .layernorm import *
from .instancenorm import *
from .residual import *

CUSTOM_OBJECTS = {
    "InstanceNormalization": InstanceNormalization,
    "LayerNormalization": LayerNormalization,
}
