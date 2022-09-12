"""
base_search
***********

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

Basis of search methods in PyCharc - this is the minimum set of things PyCharc's core interacts with
"""
from __future__ import annotations

from dataclasses import dataclass, field
from abc import abstractmethod, abstractproperty
from typing import Hashable, Sequence, Dict

import numpy.typing as npt
import numpy as np

from ..system import System 
from ..metrics import get_metrics, MetricSpec

@dataclass
class BaseItem:
    params: Hashable

    @abstractproperty
    def system(self) -> System: ...


@dataclass  # type: ignore  # Mypy bug until release with commit #13398
class BaseSearch:
    metric_spec: MetricSpec

    population_metrics: Dict[Hashable, npt.NDArray[np.floating]] = field(default_factory=dict, init=False)

    def add_to_db(self, items: Sequence[BaseItem]):
        """Add a set of individuals to the database if required"""
        for item in items:
            if item.params not in self.population_metrics:
                metrics = get_metrics(self.metric_spec, item.system)
                self.population_metrics[item.params] = metrics

    @abstractmethod
    def run(self) -> None: ...
