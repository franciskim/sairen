API Documentation
=================

.. module:: sairen

.. autoclass:: MarketEnv
   :members:  action_space, flatten, finish_on_next_step, info, _reset, _step, _close
   :show-inheritance:
   :undoc-members:

.. We can't use automodule because we don't want :members: for namedtuples, since the auto docs are pretty useless.

.. autoclass:: Obs
   :exclude-members: __init__, __new__, __class__


