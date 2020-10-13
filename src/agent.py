import os

import tensorflow as tf
from tensorflow.python.framework import tensor_spec

from gfootball.env import observation_preprocessing
import numpy as np

from .utils import _pack_named_sequence
from .TimeStep import TimeStep


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

policy = tf.saved_model.load('policy')


def main(obs):
    players = obs.controlled_players
    obs = obs['players_raw']

    state = observation_preprocessing.generate_smm(obs)

    time_step = _pack_named_sequence([[0], [0], [0], state],
                                     time_step_spec,
                                     tf.TensorShape((1,)))

    action = policy.get_action(time_step)

    return [int(action)] * players