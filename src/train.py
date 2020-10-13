from tf_agents.environments import suite_gym
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import tensorflow as tf

from .utils import collect_data, train_loop
from .FootballEnvironment import FootballEnvironment
from .DeepQNetwork import DeepQNetwork

from tensorflow.python.framework import tensor_spec

AUTO = tf.data.experimental.AUTOTUNE

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    strategy = tf.distribute.OneDeviceStrategy("/CPU:0")


def main():
    num_episodes = 5
    steps_per_episode = 3001

    gamma = 0.99
    epsilon = 0.7

    initial_collect_steps = 1500
    collect_steps_per_iteration = 15
    replay_buffer_max_length = 25000

    batch_size = 128
    learning_rate = 1e-3

    train_py_env = suite_gym.load("GFootball-11_vs_11_kaggle-SMM-v0")
    train_env = FootballEnvironment(train_py_env)

    eval_py_env = suite_gym.load("GFootball-11_vs_11_kaggle-SMM-v0")
    eval_env = FootballEnvironment(eval_py_env)

    with strategy.scope():
        action_q_net = DeepQNetwork(
            input_tensor_spec=train_env.observation_spec(),
            action_spec=train_env.action_spec(),
        )

        target_q_net = DeepQNetwork(
            input_tensor_spec=train_env.observation_spec(),
            action_spec=train_env.action_spec(),
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            gamma=gamma,
            epsilon_greedy=epsilon,
            q_network=action_q_net,
            target_q_network=target_q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter
        )

        agent.initialize()

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length
    )

    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=AUTO,
        sample_batch_size=batch_size,
        num_steps=2
    )

    dataset = dataset.prefetch(AUTO)
    dataset = strategy.experimental_distribute_dataset(dataset)

    iterator = iter(dataset)

    agent.train_step_counter.assign(0)

    train_loop(strategy,
               iterator,
               train_env,
               agent,
               replay_buffer,
               collect_steps_per_iteration,
               num_episodes,
               steps_per_episode)

    observation = tensor_spec.BoundedTensorSpec(
        shape=(1, 72, 96, 4),
        dtype=np.float32,
        name='observation',
        minimum=0.,
        maximum=255.,
    )

    reward = tf.TensorSpec(
        shape=(1,),
        dtype=np.float32,
        name='reward'
    )

    step_type_spec = tf.TensorSpec(
        shape=(1,),
        dtype=np.int32,
        name='step_type'
    )

    discount_spec = tensor_spec.BoundedTensorSpec(
        shape=(1,),
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

    @common.function
    def get_action(time_step):
        return agent.policy.action(time_step, ()).action

    setattr(agent.policy, 'get_action', get_action)
    act = agent.policy.get_action.get_concrete_function(time_step_spec)

    tf.saved_model.save(agent.policy,
                        'policy',
                        {'get_action': act})


if __name__ == '__main__':
    main()
