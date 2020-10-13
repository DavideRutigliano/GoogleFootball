from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import tensorflow as tf
from tqdm.auto import tqdm


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


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


def compute_avg_return(environment, policy, num_steps=3001, num_episodes=1):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        for _ in range(num_steps):
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


@common.function
def train_step(agent, experience, strategy):
    def replicated_train_step(experience):
        return agent.train(experience).loss

    per_replica_losses = strategy.run(
        replicated_train_step, args=(experience,)
    )

    return strategy.reduce(
        tf.distribute.ReduceOp.MEAN,
        per_replica_losses, axis=None
    )


def train_loop(strategy,
               iterator,
               train_env,
               agent,
               replay_buffer,
               collect_steps_per_iteration,
               num_episodes=1,
               steps_per_episode=100):

    for step in tqdm(range(num_episodes),
                     'Episodes',
                     num_episodes):
        train_loss = 0

        for _ in tqdm(range(steps_per_episode),
                      'Step {0}'.format(step + 1),
                      steps_per_episode):
            collect_data(train_env,
                         agent.collect_policy,
                         replay_buffer,
                         collect_steps_per_iteration)

            with strategy.scope():
                experience, _ = next(iterator)
                train_loss += train_step(agent, experience, strategy)

        print('step {0} - loss: {1:.2f}'.format(step + 1, train_loss))