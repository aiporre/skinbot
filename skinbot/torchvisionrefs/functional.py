import math
import numbers
import warnings
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

# from ..utils import _log_api_usage_once
from . import functional_pil as F_pil, functional_tensor as F_t

def get_dimensions(img: Tensor) -> List[int]:
    """Returns the dimensions of an image as [channels, height, width].
    Args:
        img (PIL Image or Tensor): The image to be checked.
    Returns:
        List[int]: The image dimensions.
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(get_dimensions)
    if isinstance(img, torch.Tensor):
        return F_t.get_dimensions(img)

    return F_pil.get_dimensions(img)