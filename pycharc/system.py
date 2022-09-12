"""
system
******

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022

System abstraction for PyCharc. A "System" is the interface between PyCharc and the underlying
reservoir.

A user should subclass System to provide the run_one method, as well as any other code required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Perceptron  # type: ignore  # TODO: write stubs
from sklearn.multioutput import MultiOutputClassifier  # type: ignore  # TODO: write stubs

from abc import ABCMeta, abstractmethod

SystemRunnable = Callable[[Sequence[float | int]], Sequence[float | int]]
Preprocessor = Callable[[np.ndarray], np.ndarray]

@dataclass  # type: ignore  # Mypy bug until release with commit #13398
class System(metaclass=ABCMeta):
    """Descriptor of the system to PyCharc. Requires some metadata """
    input_units: int
    output_units: int
    hidden_units: int
    wash_out: int
    discrete: bool = False
    concatenate_inputs: bool = False
    train_function: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]] = field(init=False, default=lambda x: x)
    total_units: int = field(init=False)

    def __post_init__(self):
        self.total_units = self.input_units + self.output_units + self.hidden_units

    def train(self, train_input: npt.NDArray[np.floating], train_output: npt.NDArray[np.floating]) -> None:
        """
        Trains the model. By default uses a multi output perceptron, but can be overridden as required.
        """
        untrained_output = self.run_untrained(train_input)
        perceptron = MultiOutputClassifier(Perceptron()).fit(untrained_output, train_output)
        self.train_function = perceptron.predict

    def reset(self) -> None:
        """Resets any state of the system. By default this does nothing."""

    def preprocess(self, input: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Performs any preprocessing (e.g. custom scaling) that the system might want on its data.
        Note that this is for input *data* and is exposed to PyCharc. If the system requires intrinsic
        processing (e.g. encoding the input into a carrier wave), this is better accomplished by
        processing the input data inside the run method (as the specifics of the system should not
        be visible to PyCharc)
        """
        return input

    @abstractmethod
    def run_one(self, input: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Runs a single input through the system and returns the output.
        This method must be overridden for a specific system."""

    def run_untrained(self, inputs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Runs an array of inputs through the system and returns the raw output."""
        output = np.zeros([inputs.shape[0], self.output_units])
        for index, inp in enumerate(inputs):
            output[index] = self.run_one(inp)
        if self.concatenate_inputs:
            output = np.concatenate([output, inputs])
        return output

    def run(self, inputs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Runs an array of inputs through the system, applies the results of training,
        and returns the output."""
        return self.train_function(self.run_untrained(inputs))


