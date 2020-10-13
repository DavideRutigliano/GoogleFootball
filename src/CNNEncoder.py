from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils

import tensorflow as tf


class CNNEncoder(network.Network):

    def __init__(self,
                 input_tensor_spec,
                 batch_squash=True,
                 **kwargs):

        self.model = self._build_model(input_tensor_spec.shape)
        self._batch_squash = batch_squash

        super(CNNEncoder, self).__init__(
            input_tensor_spec, **kwargs
        )

    @staticmethod
    def _build_model(input_shape):

        inputs = tf.keras.layers.Input(input_shape)

        out = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
        out = tf.keras.layers.Conv2D(32, 3, activation='relu')(out)
        out = tf.keras.layers.MaxPooling2D()(out)

        out = tf.keras.layers.Flatten()(out)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[out])

        return model

    def call(self, observation, step_type=None, network_state=(), training=False):

        if self._batch_squash:
            outer_rank = nest_utils.get_outer_rank(observation, self.input_tensor_spec)
            batch_squash = utils.BatchSquash(outer_rank)
            observation = tf.nest.map_structure(batch_squash.flatten, observation)

        states = self.model(observation, training=training)

        if self._batch_squash:
            states = tf.nest.map_structure(batch_squash.unflatten, states)

        return states, network_state
