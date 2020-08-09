import os

import numpy as np
import torch
from torch.nn import functional
from torch.optim import Adam

from architectures.gaussian_policy import GaussianPolicy
from architectures.utils import polyak_update


class GAIL:
    def __init__(self):
        pass