#!/usr/bin/env python
"""
Deep Deterministic Policy Gradient example using Keras-RL.
"""

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, BatchNormalization, GaussianDropout
from keras.layers.merge import concatenate
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from sairen import MarketEnv, xform


EPISODES = 16 * 60
STEPS_PER_EPISODE = 60
WARMUP_EPISODES = 10


def main():
    """Create environment, build models, train."""
    env = MarketEnv(("ES", "FUT", "GLOBEX", "USD"), obs_xform=xform.Basic(30, 4), episode_steps=STEPS_PER_EPISODE, client_id=3)
    #env = MarketEnv(("EUR", "CASH", "IDEALPRO", "USD"), max_quantity=20000, quantity_increment=20000, obs_xform=xform.Basic(30, 4), episode_steps=STEPS_PER_EPISODE, client_id=5, afterhours=False)
    obs_size = np.product(env.observation_space.shape)

    # Actor model
    dropout = 0.1
    actor = Sequential([
        Flatten(input_shape=(1,) + env.observation_space.shape),
        BatchNormalization(),
        Dense(obs_size, activation='relu'),
        GaussianDropout(dropout),
        BatchNormalization(),
        Dense(obs_size, activation='relu'),
        GaussianDropout(dropout),
        BatchNormalization(),
        Dense(obs_size, activation='relu'),
        GaussianDropout(dropout),
        BatchNormalization(),
        Dense(1, activation='tanh'),
    ])
    print('Actor model')
    actor.summary()

    action_input = Input(shape=(1,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = BatchNormalization()(x)
    x = Dense(obs_size + 1, activation='relu')(x)
    x = GaussianDropout(dropout)(x)
    x = Dense(obs_size + 1, activation='relu')(x)
    x = GaussianDropout(dropout)(x)
    x = Dense(obs_size + 1, activation='relu')(x)
    x = GaussianDropout(dropout)(x)
    x = Dense(obs_size + 1, activation='relu')(x)
    x = GaussianDropout(dropout)(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print('\nCritic Model')
    critic.summary()

    memory = SequentialMemory(limit=EPISODES * STEPS_PER_EPISODE, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.5, mu=0., sigma=.5)
    agent = DDPGAgent(nb_actions=1, actor=actor, critic=critic, critic_action_input=action_input, memory=memory, nb_steps_warmup_critic=STEPS_PER_EPISODE * WARMUP_EPISODES, nb_steps_warmup_actor=STEPS_PER_EPISODE * WARMUP_EPISODES, random_process=random_process, gamma=0.95, target_model_update=0.01)
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
