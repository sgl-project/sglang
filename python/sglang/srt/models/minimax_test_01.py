import copy 
import math 
from collections.abc import Iterable 
from typing import Optional, Union 

import regex as re 
import torch 
import torch.distributed 
import torch.nn.functional as F 
from einops import rearrange
from torch import nn
from transformers.configuration_utils import PretrainedConfig



def replace_weight_name(
    name: str, 
    key: Optional[str]=None, 
    to: Optional[str]=None,
    count: Optional[int]=None,
    #prefix is not used (may consider removing it)
    prefix: Optional[str]=None,
) -> str: 
    name = name.replace(key, to) if count is not None else \
        name.replace(key, to, count) 
    return name

def wegith_loader_