#!/usr/bin/env python
"""Simple Sairen trading example using a random agent."""
from sairen.env import MarketEnv


def main():
    """Create a market environment, instantiate a random agent, and run the agent for one episode."""
    env = MarketEnv("AAPL", obs_type='bar', obs_size=1, episode_steps=20)           # Apple stock, 1-second OHLCV bars.
    agent = RandomAgent(env.action_space)   # Actions are continuous from -1 = go short to +1 = go long.  0 is go flat.
    observation = env.reset()       # A bar observation is a numpy float array of [timestamp, open, high, low, close, volume, open_interest]
    done = False
    total_reward = 0.0              # Reward is the profit realized when a trade closes
    while not done:
        env.render()                # Action is a float where -1, 0, 1 set the (absolute) target position to short, flat, or long respectively
        observation, reward, done, info = env.step(agent.act(observation))
        total_reward += reward

    print('\nTotal profit: {:.2f}'.format(total_reward))        # Sairen will automatically (try to) cancel open orders and close positions on exit


class RandomAgent:
    """Agent that randomly samples the action space."""
    def __init__(self, action_space):
        """:param gym.Space action_space: The Space to sample from."""
        self.action_space = action_space

    def act(self, observation):
        """:Return: a random action from the action space."""
        return self.action_space.sample()       # Here the observation is ignored, but a less-random agent would want it.


if __name__ == "__main__":
    main()
