Observation Transforms
======================

.. currentmodule:: sairen

.. note::

   **TL;DR**: Pass a callable to :class:`MarkenEnv` as ``obs_xform``, give it an ``observation_space`` field; raw observation arrays in,
   cooked arrays out, or ``None`` if you don't have enough data yet; be sure to handle ``NaN``.

As powerful as deep learning has become, it is unlikely to be able to make sense of the essentially random numbers that
are raw market data.  Even if it is something as simple as feeding your algorithm the last N bars instead of one, you
will probably want to do some manual feature engineering.  For example, you may want to feed your algorithm a moving
average of prices, or keep other state.  Or you may want to get fancy and learn an entire feature extraction pipeline.
Since Sairen does not have backtesting, as you tweak your features you will probably want to replay previous raw data
with new transformations instead of starting from scratch.  You may want to share input transformations and combine
them.  In addition, many agents need to know the shape of the transformed observations, and this may depend on some
parameters, so you will want to compute the shape along with transformation.  For all these reasons and more, Sairen
provides a standard way to transform raw market data into the observations your agent sees.

At its simplest, an observation transform ("xform") can be a function that takes a numpy array and returns a transformed
version::

    def double(obs):
        return 2 * obs

You tell :class:`MarketEnv` to use it with the ``obs_xform`` argument::

    env = MarketEnv("AAPL", obs_xform=double)
    obs = env.reset()       # All quote values doubled.  Not terribly useful.

And then the observations returned from :meth:`step` will be transformed.

Many transformations will have some state (e.g., previous bars) and parameters (e.g., how many previous bars to store).
You can do this with a callable object::

    class AverageXform:
        """Average each value in the the last `lookback` observations."""
        def __init__(self, lookback):
            self.q = deque(maxlen=lookback)

        def __call__(self, obs):
            self.q.append(obs)
            return np.array(self.q).mean(axis=0)

Since your object is callable, you can test it just by calling it with some data:

    >>> xform = AverageXform(2)
    >>> xform([1,2,3])
    array([ 1.,  2.,  3.])
    >>> xform([4,5,6])
    array([ 2.5,  3.5,  4.5])

Most transformations will want to change the shape of the observation array, and a Gym environment needs to
know its observation shape.  You tell it by giving your transform object an ``observation_space`` attribute,
which should be a `gym.Space <https://gym.openai.com/docs#spaces>`__ object::

    from gym.spaces import Box
    from sairen import Quote

    class QueueXform:
        """Stack the last `lookback` observations into a 2D array."""
        def __init__(self, lookback):
            self.q = deque(maxlen=lookback)
            self.observation_space = Box(low=0, high=1e10, shape=(lookback, len(Quote._fields)))

        def __call__(self, obs):
            self.q.append(obs)
            return np.array(self.q)

::

    >>> xform = QueueXform(2)
    >>> xform([1,2,3])
    array([[1, 2, 3]])
    >>> xform([4,5,6])
    array([[1, 2, 3],
           [4, 5, 6]])

Alas, something is wrong: the first output observation has the wrong shape, because we did not yet have enough data
to create a full observation.

If your transform doesn't have enough data to build a full observation yet, just return ``None``::

    from gym.spaces import Box
    from sairen import Quote

    class QueueXform:
        """Stack the last `lookback` observations into a 2D array."""
        def __init__(self, lookback):
            self.q = deque(maxlen=lookback)
            self.observation_space = Box(low=0, high=1e10, shape=(lookback, len(Quote._fields)))

        def __call__(self, obs):
            self.q.append(obs)
            if len(self.q) < self.observation_space.shape[0]:
                return None
            else:
                return np.array(self.q)

::

    >>> xform = QueueXform(2)
    >>> xform([1,2,3])
    >>> xform([4,5,6])
    array([[1, 2, 3],
           [4, 5, 6]])

It's also very common for raw observations to contain ``NaN`` values, especially in the first few observations, so be sure
to handle those.