Observations and Actions
========================

.. currentmodule:: sairen

Observations
------------
Sairen's :class:`MarketEnv` provides observations as numpy arrays of market data. There are three types of observations,
which you choose with the :class:`MarketEnv` ``obs_type`` parameter: ``bar`` (the default), ``quote``, and ``tick``.
For the first two, the default period is 1 second, and can be changed with the ``obs_size`` parameter.  Ticks are
generated as values change, which can be many times per second, rather than on a schedule, Each type has a namedtuple
that can wrap the raw observation arrays for convenience; their documentation below explains the values in the order
they occur.

.. autoclass:: Quote
   :noindex:

.. autoclass:: Bar
   :noindex:

.. autoclass:: Tick
   :noindex:

Actions
-------
.. autoattribute:: MarketEnv.action_space
   :noindex: