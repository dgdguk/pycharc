"""
sources
*******

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

Package containing random sources.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable

from .random import random_source as random_source

SourceType = Callable[[int, int], npt.NDArray[np.floating]]