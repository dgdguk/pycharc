"""
microbial_ga
************

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

Microbial Genetic Algorithm search method for PyCharc
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Hashable, Any, Callable, Sequence, List, Dict

import numpy as np
import numpy.typing as npt

from ..metrics.registry import get_metrics

from .base_ga import GAIndividual, BaseGA

@dataclass
class KNNFitness:
    k_value: int

    def __call__(self, population: List[GAIndividual], metrics: Dict[Hashable, npt.NDArray[np.floating]], entry: int) -> float:
        """Calculate the KNN fitness function for the given k value"""
        assert len(population) > self.k_value
        individual = population[entry]
        individual_metrics = metrics[individual.params]
        distances = [
            (np.linalg.norm(individual_metrics - metrics[other.params]),
            other) for other in population if other is not individual]
        distances.sort()
        return float(distances[self.k_value][0])
        

@dataclass
class MicrobialGA(BaseGA):
    
    deme_rate: float = 0.2
    
    fitness_function: Callable[[List[GAIndividual], Dict[Hashable, npt.NDArray[np.floating]], int], float] = KNNFitness(10)

    def __post_init__(self):
        """Calculate the deme range from parameters"""
        self.deme_range = int(self.population_size / self.deme_rate + 0.5)

    def round(self):
        """Run one round of the Microbial GA"""
        index_1, index_2 = -1, -1
        while index_2 == index_1:
            index_1 = random.randint(0, self.population_size - 1)
            index_2 = (index_1 + random.randint(1, self.deme_range)) % self.population_size
        fitness_1 = self.fitness_function(self.population, self.population_metrics, index_1)
        fitness_2 = self.fitness_function(self.population, self.population_metrics, index_2)
        winner_index, loser_index = (index_1, index_2) if fitness_1 > fitness_2 else (index_2, index_1)
        winner, loser = self.population[winner_index], self.population[loser_index]
        offspring = self.combine_func(winner, loser, self.recombination_rate)
        offspring = self.mutate_func(offspring, self.mutation_rate)
        self.population[loser_index] = offspring
        self.new_individuals.append(offspring)
