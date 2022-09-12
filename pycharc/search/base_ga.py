"""
base_ga
*******

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

Basis for GA search methods
"""

from __future__ import annotations

from abc import abstractproperty, abstractmethod, ABCMeta
from typing import Hashable, TYPE_CHECKING, Callable, Sequence, Tuple, List, Dict, Union
import random
from dataclasses import dataclass, field
from .base_search import BaseSearch, BaseItem

if TYPE_CHECKING:
    from ..system import System


class GAParameterSpec:
    @abstractmethod
    def new_random_value(self) -> Hashable: ...

@dataclass 
class DiscreteGAParameterSpec(GAParameterSpec):
    possible_values: Sequence[Hashable]

    def new_random_value(self) -> Hashable:
        return random.choice(self.possible_values)

@dataclass
class FloatingGAParameterSpec(GAParameterSpec):
    min_value: float
    max_value: float
    range: float = field(init=False)

    def __post_init__(self):
        if self.min_value > self.max_value: raise ValueError(f'FloatingGAParameter with {self.min_value=} > {self.max_value=}')
        self.range = self.max_value - self.min_value

    def new_random_value(self) -> float:
        return self.range * random.random() + self.min_value

@dataclass
class IntGAParameterSpec(GAParameterSpec):
    min_value: int
    max_value: int
    def __post_init__(self):
        if self.min_value > self.max_value: raise ValueError(f'IntGAParameter with {self.min_value=} > {self.max_value=}')

    def new_random_value(self) -> int:
        return random.randint(self.min_value, self.max_value)


@dataclass  # type: ignore  # Mypy bug until release with commit #13398
class GAIndividual(BaseItem):  
    """Individual for a GA. Contains a Tuple of current GAParameters and a property containing the current
    system.
    """
    params: Tuple[Hashable, ...]
    params_spec_dict: Dict[str, GAParameterSpec]
    params_spec: Tuple[GAParameterSpec, ...] = field(init=False)
    params_names: Dict[str, int] = field(init=False)

    def __post_init__(self):
        if len(self.params) != len(self.params_spec_dict):
            raise ValueError('GAIndividual with {len(self.params)=} != {len(self.params_spec)=}')
        self.params_spec = tuple(self.params_spec_dict.values())
        self.params_names = {key: indx for indx, key in enumerate(self.params_spec_dict.keys())}

    def __getitem__(self, key: str):
        return self.params[self.params_names[key]]

    @abstractproperty
    def system(self) -> System: ...

    def reset_system(self):
        """Reset the system, if required"""
        self.system.reset()


def combine_noop(major: GAIndividual, minor: GAIndividual, rate: float) -> GAIndividual:
    """Combination noop"""
    return minor

def mutate_noop(subject: GAIndividual, rate: float) -> GAIndividual:
    """Mutate noop"""
    return subject

def combine_piecewise(major: GAIndividual, minor: GAIndividual, rate: float) -> GAIndividual:
    """Piecewise combination; the minor individual is changed by combining it with the major individual"""
    new_params = tuple(major_param if random.random() < rate else minor_param for major_param, minor_param in zip(major.params, minor.params))
    minor.params = new_params
    minor.reset_system()
    return minor

def mutate_piecewise(subject: GAIndividual, rate: float) -> GAIndividual:
    """Piecewise mutation: the subject is changed by mutating its parts"""
    new_params = tuple(spec.new_random_value() if random.random() < rate else param for param, spec in zip(subject.params, subject.params_spec))
    subject.params = new_params
    subject.reset_system()
    return subject


def piecewise_params(spec: Union[Dict[str, GAParameterSpec], Sequence[GAParameterSpec]]) -> Tuple[Hashable, ...]:
    if isinstance(spec, dict):
        return tuple(x.new_random_value() for x in spec.values())
    else:
        return tuple(x.new_random_value() for x in spec)


def piecewise_individual(spec: Dict[str, GAParameterSpec], cls: Callable[[Tuple[Hashable, ...], Dict[str, GAParameterSpec]], GAIndividual]) -> GAIndividual:
    return cls(piecewise_params(spec), spec)


@dataclass  # type: ignore  # Mypy bug until release with commit #13398
class BaseGA(BaseSearch):
    generator: Callable[[], GAIndividual]

    number_of_tests: int = 1
    population_size: int = 100
    generations: int = 100
    mutation_rate: float = 0.1
    recombination_rate: float = 0.5

    combine_func: Callable[[GAIndividual, GAIndividual, float], GAIndividual] = combine_piecewise
    mutate_func: Callable[[GAIndividual, float], GAIndividual] = mutate_piecewise

    population: List[GAIndividual] = field(default_factory=list, init=False)
    new_individuals: List[GAIndividual] = field(default_factory=list, init=False)

    @abstractmethod
    def round(self) -> None:
        """Run a round of the GA. This must be overridden by a subsclass to define what a round is"""

    def make_population(self):
        """Make the population by calling the individual generator function"""
        self.population = [self.generator() for _ in range(self.population_size)]

    def run(self):
        """Run the Microbial GA"""
        self.make_population()
        self.add_to_db(self.population)

        for _ in range(self.generations):
            self.new_individuals = []
            self.round()
            self.add_to_db(self.new_individuals)
