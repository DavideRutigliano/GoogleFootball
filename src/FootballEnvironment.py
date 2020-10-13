from tf_agents import specs
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts

import numpy as np
import tensorflow as tf

from .utils import _pack_named_sequence


class FootballEnvironment(tf_py_environment.TFPyEnvironment):

    def __init__(self, env, **kwargs):
        self.observation = specs.BoundedTensorSpec(
            shape=(72, 96, 4),
            dtype=np.float32,
            name='observation',
            minimum=0.,
            maximum=255.,
        )

        self.reward = specs.TensorSpec(
            shape=(),
            dtype=np.float32,
            name='reward'
        )

        step_type_spec = specs.TensorSpec(
            shape=(),
            dtype=np.int32,
            name='step_type'
        )

        discount_spec = specs.BoundedTensorSpec(
            shape=(),
            dtype=np.float32,
            minimum=0.,
            maximum=1.,
            name='discount'
        )

        self.time_step = ts.TimeStep(
            observation=self.observation,
            reward=self.reward,
            step_type=step_type_spec,
            discount=discount_spec
        )

        self.action = specs.BoundedTensorSpec(
            shape=(),
            dtype=np.int32,
            name='action',
            minimum=0,
            maximum=18
        )

        self._time_step_dtypes = [
            s.dtype for s in tf.nest.flatten(self.time_step_spec())
        ]

        super(FootballEnvironment, self).__init__(env, **kwargs)

    def _time_step_from_numpy_function_outputs(self, outputs):
        batch_shape = () if not self.batched else (self.batch_size,)
        batch_shape = tf.TensorShape(batch_shape)
        time_step = _pack_named_sequence(outputs,
                                         self.time_step_spec(),
                                         batch_shape)
        return time_step

    def observation_spec(self):
        return self.observation

    def reward_spec(self):
        return self.reward

    def time_step_spec(self):
        return self.time_step

    def action_spec(self):
        return self.action
