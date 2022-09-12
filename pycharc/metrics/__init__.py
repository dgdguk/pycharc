"""
metrics
*******

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

Package containing the metrics of PyCharc and a registry of metrics

To add a new metric so that it has a symbol attached to it, use the register function e.g.

@register("my_new_metric")
def my_new_metric(system: System) -> float: ...
"""
from __future__ import annotations

from . import generalisation_rank
from . import kernel_rank
from . import linearmc

from .registry import (
    get_metrics as get_metrics,
    register as register,
    MetricSpec as MetricSpec
)
