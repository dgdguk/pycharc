"""
generalisation_rank
*******************

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import matrix_rank
from ..system import System
from ..sources import random_source, SourceType
from .registry import register


@register('GR')
def generalisation_rank(system: System, source: SourceType=random_source):
    """Generalisation Rank, as implemented in CHARC"""
    num_timesteps = int(system.total_units * 1.5 + 0.5) + system.wash_out
    n_input_units = system.input_units
    input_signal = 0.5 + 0.1*source(num_timesteps, n_input_units)-0.05
    input_signal = system.preprocess(input_signal)
    output = system.run(input_signal)
    np.nan_to_num(output, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    gr = matrix_rank(output)
    return gr
