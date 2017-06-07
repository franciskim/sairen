#!/usr/bin/env python
"""
Simple OpenAI Gym environment recording and playback.
"""

import logging

import numpy as np
import gym
from gym.spaces import Box

# TODO: RecordEnv environment that pushes an iterable of (obs, action, reward, info(?)) tuples to a writer


class PlaybackEnv(gym.Env):
    """Playback a sequence of observations and ignore actions."""
    action_space = Box(-1, 1, shape=(1,))
    metadata = {'render.modes': ['human']}

    def __init__(self, observations):
        """:param iter(iter(array)) observations: Iterable over episodes, each of which is an iterable over steps,
        each of which is a numpy array containing an observation."""
        self.observations = observations
        self.done = True
        self.episodes = iter(observations)
        self.episode = -1
        self.steps = iter(next(self.episodes))
        self.step_num = 1
        self.first_obs = next(self.steps)
        self.obs = None
        self.next_obs = None
        self.observation_space = Box(-np.inf, np.inf, shape=self.first_obs.shape)
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.log.debug('Playback')

    def _reset(self):
        """Called before the start of every episode."""
        self.log.debug('EP {} RESET'.format(self.episode + 1))
        self.done = False
        if self.first_obs is not None:      # Very first ever step
            self.obs = self.first_obs
            self.first_obs = None
        else:
            self.steps = next(self.episodes, None)
            if self.steps is None:
                raise ValueError('No more episodes to play back')
            self.steps = iter(self.steps)
            self.step_num = 1
            self.obs = next(self.steps)
        try:
            self.next_obs = next(self.steps)
        except StopIteration:
            self.next_obs = None
            self.done = True
        self.episode += 1
        self.log.debug('EP {} STEP 0 RESET: {}'.format(self.episode, self.obs))
        return self.obs

    def _step(self, action):
        """:Return: (observation, reward, done, info) for each step."""
        self.log.debug('EP {} STEP {} ACTION: {}'.format(self.episode, self.step_num, action))
        self.obs = self.next_obs
        if self.obs is None or self.done:
            raise ValueError('Must call reset() before step()')
        try:
            self.next_obs = next(self.steps)
        except StopIteration:
            self.next_obs = None
            self.done = True
        self.step_num += 1
        return self.obs, 0, self.done, {}

    def _render(self, mode='human', close=False):
        if mode == 'human':
            if not close:
                print('EP {} STEP {} OBS {}'.format(self.episode, self.step_num - 1, self.obs))
