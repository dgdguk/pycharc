"""
kernel_rank
***********

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

"""
from __future__ import annotations

import numpy as np
from numpy.linalg import matrix_rank
from .registry import register
from ..system import System
from ..sources import random_source, SourceType


@register('KR')
def kernel_rank(system: System, source: SourceType=random_source, threshold=0.1):
    # NOTE: Ask Susan: Do we want initial wash_out in KR/GR?
    num_timesteps = int(system.total_units * 1.5 + 0.5) + system.wash_out
    n_input_units = system.input_units
    input_signal = 2*source(num_timesteps, n_input_units)-1
    input_signal = system.preprocess(input_signal)
    output = system.run(input_signal)
    np.nan_to_num(output, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    kr = matrix_rank(output, tol=threshold)
    return kr

