from typing import Callable, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForConditionalGeneration
)

EntryClass = Gemma3ForConditionalGeneration
