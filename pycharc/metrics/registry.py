"""
registry
********

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022
"""
from __future__ import annotations

from typing import Hashable, Dict, Any, TYPE_CHECKING, Union, Tuple, Iterable, Callable
import collections
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..system import System


# Note: Mypy doesn't support the better way of doing this specification, using a ParamSpec.
# As such, any metric function won't get type checked.
MetricFunction = Callable[..., float]

__REGISTRY: Dict[Hashable, MetricFunction] = {}

MetricSpec = Iterable[Union[Tuple[Union[Hashable, MetricFunction], Iterable[Any]], Hashable]]

def register(name: Hashable) -> Callable[[MetricFunction], MetricFunction]:
    """Register a function as a metric with the given name"""
    def f(func: MetricFunction) -> MetricFunction:
        if name in __REGISTRY:
            prior = __REGISTRY[name]
            raise ValueError(f'Tried to register {func.__module__}.{func.__name__} as {name}, but {name} already used by {prior.__module__}.{prior.__name__}')
        __REGISTRY[name] = func
        return func
    return f


def get_metric_function(name: Hashable) -> MetricFunction:
    """Retrieves a metric function from the registry"""
    if name in __REGISTRY:
        return __REGISTRY[name]
    else:
        raise ValueError(f'No metric function with {name} is registered')


def get_metrics(metricspec: MetricSpec, system: System) -> npt.NDArray[np.floating]:
    """Applies a MetricSpec (a list containing items which are either names in the registry or a tuple containing 
    a name or callable metric followed by a list of additional arguments) to a given System. Returns the metrics
    requested."""
    metrics = []
    for spec in metricspec:
        if isinstance(spec, (list, tuple)):
            if len(spec) == 2:
                metric, args = spec
            else:
                raise ValueError(f'Error with metric specification: {spec}')
        else:
            metric = spec
            args = []

        if callable(metric):
            metric_func = metric
        elif isinstance(metric, collections.abc.Hashable):
            metric_func = get_metric_function(metric)
        else:
            raise ValueError(f'Error with metric function spec: {metric}')
        metrics.append(metric_func(system, *args))
    return np.asarray(metrics)
