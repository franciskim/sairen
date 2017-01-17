#!/usr/bin/env python
"""
Example using the Cross-Entropy Method and deep learning with Keras RL.
"""
import json
from functools import reduce
import operator
from datetime import datetime

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from rl.agents.cem import CEMAgent
from rl.callbacks import FileLogger, Callback
from rl.memory import EpisodeParameterMemory
from rl.core import Processor
from sairen.env import MarketEnv
from sairen import xform

EPISODES = 1000
STEPS_PER_EPISODE = 60
WARMUMP_EPISODES = 30
TRAIN_INTERVAL_EPISODES = 5

# EPISODES = 3
# STEPS_PER_EPISODE = 5
# WARMUMP_EPISODES = 1
# TRAIN_INTERVAL_EPISODES = 1


def main():
    """Build model and train on environment."""
    env = MarketEnv(("ES", "FUT", "GLOBEX", "USD"), obs_xform=xform.BinaryDelta(3), episode_steps=STEPS_PER_EPISODE, client_id=3)
    #env = MarketEnv(("AAPL", "STK", "SMART", "USD"), obs_xform=xform.BinaryDelta(3), episode_steps=STEPS_PER_EPISODE, client_id=4)
    nb_actions = 3      # Keras-RL CEM is a discrete agent

    # Option 1 : Simple model
    model = Sequential([
        Flatten(input_shape=(1,) + env.observation_space.shape),
        Dense(nb_actions),
        Activation('softmax')
    ])

    # Option 2: deep network
    # hidden_nodes = reduce(operator.imul, env.observation_space.shape, 1)
    # model = Sequential([
    #     Flatten(input_shape=(1,) + env.observation_space.shape),
    #     Dense(hidden_nodes),
    #     Activation('relu'),
    #     Dense(hidden_nodes),
    #     Activation('relu'),
    #     Dense(hidden_nodes),
    #     Activation('relu'),
    #     Dense(nb_actions),
    #     Activation('softmax')
    # ])

    print(model.summary())

    param_logger = CEMParamLogger('cem_{}_params.json'.format(env.instrument.symbol))
    callbacks = [
        param_logger,
        FileLogger('cem_{}_log.json'.format(env.instrument.symbol), interval=STEPS_PER_EPISODE)
    ]

    theta_init = param_logger.read_params()     # Start with last saved params if present
    if theta_init is not None:
        print('Starting with parameters from {}:\n{}'.format(param_logger.params_filename, theta_init))

    memory = EpisodeParameterMemory(limit=EPISODES, window_length=1)        # Remember the parameters and rewards for the last `limit` episodes.
    cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=EPISODES, nb_steps_warmup=WARMUMP_EPISODES * STEPS_PER_EPISODE, train_interval=TRAIN_INTERVAL_EPISODES, elite_frac=0.2, theta_init=theta_init, processor=DiscreteProcessor(), noise_decay_const=0, noise_ampl=0)
    """
    :param memory: Remembers the parameters and rewards for the last `limit` episodes.
    :param int batch_size: Randomly sample this many episode parameters from memory before taking the top `elite_frac` to construct the next gen parameters from.
    :param int nb_steps_warmup: Run for this many steps (total) to fill memory before training
    :param int train_interval: Train (update parameters) every this many episodes
    :param float elite_frac: Take this top fraction of the `batch_size` randomly sampled parameters from the episode memory to construct new parameters.
    """
    cem.compile()
    cem.fit(env, nb_steps=STEPS_PER_EPISODE * EPISODES, visualize=True, verbose=2, callbacks=callbacks)
    cem.save_weights('cem_{}_weights.h5f'.format(env.instrument.symbol), overwrite=True)
    #cem.test(env, nb_episodes=2, visualize=True)


class DiscreteProcessor(Processor):
    """Convert discrete actions 0, 1, 2 to -1, 0, 1 (short, flat, long)."""
    def process_action(self, action):
        assert 0 <= action <= 2
        return action - 1


class CEMParamLogger(Callback):
    """Log CEM parameters (theta) to a file after each update.

    Params are appended to the file as a single-line JSON dict with key 'theta'.
    Params are only logged every `train_interval` episodes of the CEMAgent.
    """
    JSON_COMPACT_SEPARATORS = (',', ':')

    def __init__(self, params_filename, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params_filename = params_filename

    def on_episode_end(self, episode, logs):
        if episode % self.model.train_interval == 0:        # self.model is really the CEMAgent
            data = dict(logs)
            data.update({
                'date': datetime.utcnow().replace(microsecond=0).isoformat(),
                'episode': episode,
                'theta': list(self.model.theta),
            })
            with open(self.params_filename, 'a') as paramsfile:
                json.dump(data, paramsfile, separators=self.JSON_COMPACT_SEPARATORS)
                print(file=paramsfile)

    def read_params(self):
        """:Return: a numpy array containing the last `theta` written to the log file, or None if there are none."""
        try:
            line = None
            with open(self.params_filename, 'r') as paramsfile:
                for line in paramsfile:
                    pass
            data = json.loads(line)
            theta = np.array(data.get('theta'))
            return theta
        except:     # So many things to go wrong
            return None


if __name__ == '__main__':
    main()
