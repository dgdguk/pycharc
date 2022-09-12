"""
random
******

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

Basic uniform random source
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

def random_source(timesteps: int, input_size: int) -> npt.NDArray[np.floating]:
    return np.random.random([timesteps, input_size])
