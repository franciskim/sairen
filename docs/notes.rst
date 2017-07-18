Notes!
======

Implementation working notes.

Design choices
--------------

Include current position and/or unrealized gain in observation?
  - Pros: probably much easier to learn from (high unrealized gain -> good; selling -> short position -> downticks good)
  - Cons: More agent state than environment observation, pretty useless for future agents making different decisions.

Include afterhours as constructor flag, or observation variable?  Time since open (regular? afterhours?) as obs variable?


