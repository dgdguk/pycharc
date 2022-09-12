"""
linearmc
********

:author: David Griffin
:license: GPL v3
:copyright: 2021-2022
"""
from __future__ import annotations

import numpy as np

from .registry import register

from ..system import System
from ..sources import random_source, SourceType


@register('linearMC')
def linear_memory_capacity(system: System, source: SourceType=random_source, filter: float=0.0) -> float:
    data_length = 500 + system.wash_out*2
    sequence_length = data_length // 2
    data_sequence = 2 * source(data_length + 1 + system.output_units, 1) - 1
    data_sequence = system.preprocess(data_sequence)

    if system.discrete:
        data_sequence = data_sequence - data_sequence % 1
    
    input_data = data_sequence[system.output_units:data_length+system.output_units, :]
    
    input_sequence = np.repeat(input_data, system.input_units, axis=1)
    output_sequence = np.zeros([data_length, system.output_units])

    for i in range(system.output_units):
        output_sequence[:, i] = data_sequence[i:data_length+i,:].transpose()

    train_input_sequence = input_sequence[:sequence_length]
    train_output_sequence = output_sequence[:sequence_length]

    test_input_sequence = input_sequence[sequence_length:]
    test_output_sequence = output_sequence[sequence_length:]

    system.reset()  # Reset any state of the sysem
    system.train(train_input_sequence, train_output_sequence)  # Train system
    
    predictions = system.run(test_input_sequence)  # Get predictions

    # Trim off the wash out period
    predictions = predictions[system.wash_out:]
    test_output_sequence = test_output_sequence[system.wash_out:]

    # Calculate memory capacities.

    input_variance = np.var(test_input_sequence[system.wash_out:])

    memory_capacities = []

    for i in range(system.output_units):   
        mean_output = np.mean(test_output_sequence[:, i])
        mean_predict = np.mean(predictions[:, i])
        sz = predictions.shape[0]
        covariance = ((test_output_sequence[:, i] - mean_output) * (predictions[:, i] - mean_predict) / (sz - 1)).sum()
        prediction_variance = np.var(predictions[:, i])
        memory_capacity = (covariance ** 2) / (input_variance * prediction_variance)
        if memory_capacity < filter:
            memory_capacity = 0.0
        memory_capacities.append(memory_capacity)

    return sum(memory_capacities)
