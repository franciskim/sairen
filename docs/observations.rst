Observations and Actions
========================

.. currentmodule:: sairen

Observations
------------
Sairen's :class:`MarketEnv` provides observations as numpy arrays of market data ("bars").  The default bar period is 1
second, and can be changed with the ``obs_size`` parameter.  The :class:`Bar` :obj:`namedtuple <collections.namedtuple>` can wrap raw observation arrays
to access values by name::

    bar = Bar._make(obs)
    print(bar.time, bar.open, bar.high, bar.low, bar.close, bar.volume)


The documentation below explains the values in the order they occur.

.. autoclass:: Bar
   :noindex:

If you would prefer to build bars yourself, you can also get a "bar" on every quote change from IB by using ``obs_type='tick'``.
Note these can arrive many times per second, generally too fast to reasonably act on directly since your connection latency
is generally higher than that.


Actions
-------
.. autoattribute:: MarketEnv.action_space
   :noindex: