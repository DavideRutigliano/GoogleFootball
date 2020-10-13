from tf_agents.networks import network

import tensorflow as tf

from .CNNEncoder import CNNEncoder


class DeepQNetwork(network.Network):

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='QNetwork'):
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1

        encoder = CNNEncoder(input_tensor_spec=input_tensor_spec)

        q_value_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.compat.v1.initializers.random_uniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.compat.v1.initializers.constant(-0.2),
            dtype=dtype)

        super(DeepQNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name
        )

        self._encoder = encoder
        self._q_value_layer = q_value_layer

    def call(self,
             observation,
             step_type=None,
             network_state=(),
             training=False):
        state, network_state = self._encoder(
            observation,
            step_type=step_type,
            network_state=network_state,
            training=training
        )

        q_value = self._q_value_layer(state, training=training)
        return q_value, network_state