import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_spec

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
