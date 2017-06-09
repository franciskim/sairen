#!/usr/bin/env python
"""
Deep Deterministic Policy Gradient example using Keras-RL.
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, BatchNormalization
from keras.layers.merge import concatenate
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from sairen import MarketEnv, xform

EPISODES = 16 * 60
STEPS_PER_EPISODE = 60


def main():
    """Create environment, build models, train."""
    env = MarketEnv(("ES", "FUT", "GLOBEX", "USD"), obs_xform=xform.Delta(4), episode_steps=STEPS_PER_EPISODE, client_id=3)
    assert env.action_space.shape == (1,)
    assert len(env.observation_space.shape) == 1
    obs_size = env.observation_space.shape[0]

    # Actor model
    actor = Sequential([
        Flatten(input_shape=(1, obs_size)),
        BatchNormalization(),
        Dense(obs_size, activation='relu'),
        BatchNormalization(),
        Dense(obs_size, activation='relu'),
        BatchNormalization(),
        Dense(obs_size, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='tanh'),
    ])
    print('Actor model')
    actor.summary()

    action_input = Input(shape=(1,), name='action_input')
    observation_input = Input(shape=(1, obs_size), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(obs_size + 1, activation='relu')(x)
    x = Dense(obs_size + 1, activation='relu')(x)
    x = Dense(obs_size + 1, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print('\nCritic Model')
    critic.summary()

    memory = SequentialMemory(limit=EPISODES * STEPS_PER_EPISODE, window_length=1)
    random_process = None   # OrnsteinUhlenbeckProcess(theta=.5, mu=0., sigma=.5)
    agent = DDPGAgent(nb_actions=1, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=STEPS_PER_EPISODE * 5, nb_steps_warmup_actor=STEPS_PER_EPISODE * 5, random_process=random_process, gamma=0.99, target_model_update=0.05)
    agent.compile('rmsprop', metrics=['mae'])
    weights_filename = 'ddpg_{}_weights.h5f'.format(env.instrument.symbol)
    try:
        agent.load_weights(weights_filename)
        print('Using weights from {}'.format(weights_filename))     # DDPGAgent actually uses two separate files for actor and critic derived from this filename
    except IOError:
        pass
    agent.fit(env, nb_steps=EPISODES * STEPS_PER_EPISODE, visualize=True, verbose=2, nb_max_episode_steps=STEPS_PER_EPISODE)
    agent.save_weights(weights_filename, overwrite=True)
    #agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=STEPS_PER_EPISODE)


if __name__ == '__main__':
    main()
