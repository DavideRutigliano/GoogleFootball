import os

import tensorflow as tf
from tensorflow.python.framework import tensor_spec

from gfootball.env import observation_preprocessing
import numpy as np

from typing import NamedTuple


class StepType(object):
    FIRST = np.asarray(0, dtype=np.int32)
    MID = np.asarray(1, dtype=np.int32)
    LAST = np.asarray(2, dtype=np.int32)

    def __new__(cls, value):
        if value == cls.FIRST:
            return cls.FIRST
        if value == cls.MID:
            return cls.MID
        if value == cls.LAST:
            return cls.LAST

        raise ValueError('No known conversion for `%r` into a StepType' % value)


class TimeStep(
    NamedTuple('TimeStep', [('step_type', tf.TensorSpec),
                            ('reward', tensor_spec.BoundedTensorSpec),
                            ('discount', tf.TensorSpec),
                            ('observation', tensor_spec.BoundedTensorSpec)])):
    __slots__ = ()

    def is_first(self):
        if tf.is_tensor(self.step_type):
            return tf.equal(self.step_type, StepType.FIRST)
        return np.equal(self.step_type, StepType.FIRST)

    def is_mid(self):
        if tf.is_tensor(self.step_type):
            return tf.equal(self.step_type, StepType.MID)
        return np.equal(self.step_type, StepType.MID)

    def is_last(self):
        if tf.is_tensor(self.step_type):
            return tf.equal(self.step_type, StepType.LAST)
        return np.equal(self.step_type, StepType.LAST)

    def __hash__(self):
        return hash(tuple(tf.nest.flatten(self)))


def _pack_named_sequence(flat_inputs, input_spec, batch_shape):
    named_inputs = []
    for flat_input, spec in zip(flat_inputs, tf.nest.flatten(input_spec)):
        if spec.name == 'observation':
            named_input = tf.identity(tf.cast(flat_input, spec.dtype) / 255.0, name=spec.name)
        else:
            named_input = tf.identity(tf.cast(flat_input, spec.dtype), name=spec.name)
        if not tf.executing_eagerly():
            named_input.set_shape(batch_shape.concatenate(spec.shape))
        named_inputs.append(named_input)

    nested_inputs = tf.nest.pack_sequence_as(input_spec, named_inputs)
    return nested_inputs


observation = tensor_spec.BoundedTensorSpec(
    shape=(72, 96, 4),
    dtype=np.float32,
    name='observation',
    minimum=0.,
    maximum=255.,
)

reward = tf.TensorSpec(
    shape=(),
    dtype=np.float32,
    name='reward'
)

step_type_spec = tf.TensorSpec(
    shape=(),
    dtype=np.int32,
    name='step_type'
)

discount_spec = tensor_spec.BoundedTensorSpec(
    shape=(),
    dtype=np.float32,
    minimum=0.,
    maximum=1.,
    name='discount'
)

time_step_spec = TimeStep(
    observation=observation,
    reward=reward,
    step_type=step_type_spec,
    discount=discount_spec
)

policy = tf.saved_model.load('/kaggle_simulations/agent/policy')


def main(obs):
    players = obs.controlled_players
    obs = obs['players_raw']

    state = observation_preprocessing.generate_smm(obs)

    time_step = _pack_named_sequence([[0], [0], [0], state],
                                     time_step_spec,
                                     tf.TensorShape((1,)))

    action = policy.get_action(time_step)

    return [int(action)] * players