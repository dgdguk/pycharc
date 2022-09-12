"""
esn_example
***********

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

A demonstration of PyCharc to explore the behaviour space of ESNs

Requires the optional requirements file (pip install -r requirements-optional.txt)
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

import reservoirpy.utils
from reservoirpy.node import Node
from reservoirpy.nodes import Reservoir, Ridge, Input

from typing import Dict

from pycharc.search.base_ga import (
    GAIndividual, DiscreteGAParameterSpec, FloatingGAParameterSpec, IntGAParameterSpec,
    GAParameterSpec, piecewise_individual, mutate_piecewise, combine_piecewise)
from pycharc.search.microbial_ga import MicrobialGA
from pycharc.system import System

"""
This sets reservoirpy's verbosity to 0 - otherwise we will get a lot of printouts
"""
reservoirpy.utils.VERBOSITY = 0

"""
First, we define a probably excessive number of parameters, which illustrate how to use each of the Parameter Specifications.

In practice, for the MicrobialGA, it's probably best to use a smaller number of parameters.
"""
ESNParamSpec: Dict[str, GAParameterSpec] = {
    'hidden_size': IntGAParameterSpec(5, 100),
    'number_layers': IntGAParameterSpec(1, 3),
    'nonlinearity': DiscreteGAParameterSpec(['tanh', 'relu', 'id']),
    'spectral_radius': FloatingGAParameterSpec(0.5, 0.9),
    'leaking_rate': FloatingGAParameterSpec(0.5, 1.5)
}

"""Next we define the ESNSystem representation."""

@dataclass
class ESNSystem(System):
    """Representation of an ESN system.
    
    Please note: PyTorch-ESN stealthily added a hard nVidia GPU requirement which broke my test setup, so this code
    almost certainly does not work.

    A pure CPU ESN will be added as soon as possible. In the meantime, this class should be treated as an illustrative
    example only.
    """
    spectral_radius: float = 0.9
    leaking_rate: float = 1.0
    nonlinearity: str = 'tanh'
    num_layers: int = 1

    esn: Node = field(init=False)

    def __post_init__(self):
        """Initialises the actual underlying PyTorch ESN"""
        super().__post_init__()
        input_layer = Input(self.input_units)
        reservoir = Reservoir(units=self.hidden_units, lr=self.leaking_rate, sr=self.spectral_radius, activation=self.nonlinearity)
        output = Ridge(output_dim=self.output_units, ridge=1e-6)
        self.esn = input_layer >> reservoir >> output

    def train(self, train_input: npt.NDArray[np.floating], train_output: npt.NDArray[np.floating]) -> None:
        """Overrides the train method to use PyTorch-ESN's optimised training. Normally this is not necessary
        and you can just use the default train method (perceptron-per-output)"""
        self.esn.fit(train_input, train_output, warmup=self.wash_out)
        

    def run_one(self, input: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Supplies the run_one method"""
        return self.esn.run([input])

    def run(self, inputs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Overrides the run method to use PyTorch-ESN's native methods. Again, this is normally not necessary."""
        return self.esn.run(inputs)

"""Now, we define the ESNIndividual which converts the GA representation to the ESNSystem representation,
as well as filling in some basic parameters. GAIndividual subclasses are compatible with the
default mutation/combination/initialisation operators provided by PyCharc."""

class ESNIndividual(GAIndividual):
    @property
    def system(self) -> ESNSystem:
        """Returns an ESNSystem representation by simply creating a new one."""
        return ESNSystem(
            input_units = 1,
            output_units = 1,
            hidden_units = self['hidden_size'],
            wash_out = int(1.5 * self['hidden_size']),
            discrete = False,
            concatenate_inputs=True, 
            nonlinearity=self['nonlinearity'],
            num_layers=self['number_layers']
        )

"""Finally we can define the MicrobialGA specification. You don't have to specify all of these
parameters in general - a lot have "sensible" default values."""

search = MicrobialGA(
    metric_spec=['KR', 'GR', 'linearMC'],
    generator = lambda: piecewise_individual(ESNParamSpec, lambda x, y: ESNIndividual(x, y)),
    number_of_tests=1,
    population_size=100,
    generations=100,
    mutation_rate=0.1,
    recombination_rate=0.5,
    deme_rate = 0.2,
    combine_func=combine_piecewise,
    mutate_func=mutate_piecewise
)

search.run()

"""You can now inpsect the database of individuals in search.population_metrics, which lists the
individuals encountered."""