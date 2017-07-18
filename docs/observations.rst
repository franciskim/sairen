Observations and Actions
========================

.. currentmodule:: sairen

Observations
------------
Sairen's :class:`MarketEnv` provides observations as numpy arrays of market data.  The default observation period is 1
second, and can be changed with the ``obs_size`` parameter.  The :class:`Obs` :obj:`namedtuple <collections.namedtuple>` can wrap raw observation arrays
to access values by name::

    obs = Obs._make(obs)
    print(obs.time, obs.open, obs.high, obs.low, obs.close, obs.volume)


The documentation below explains observation values in the order they occur.

.. autoclass:: Obs
   :noindex:

If you would prefer to build bars yourself, you can also get a "bar" on every quote change from IB by using ``obs_type='tick'``.
Note these can arrive many times per second, generally too fast to reasonably act on directly since your connection latency
is generally higher than the tick period.


Actions
-------
.. autoattribute:: MarketEnv.action_space
   :noindex: