Sairen - OpenAI Gym Reinforcement Learning Environment for the Stock Market
===========================================================================

Sairen (pronounced `"Siren" <https://en.wikipedia.org/wiki/Siren_(mythology)>`__) connects
artificial intelligence to the stock market. No, not in that vapid elevator pitch sense: Sairen is an
`OpenAI <https://openai.com/>`__ `Gym <https://gym.openai.com/>`__ environment for the
`Interactive Brokers <https://www.interactivebrokers.com/>`__
`API <https://interactivebrokers.github.io/tws-api/>`__. That means is it provides a standard
interface for off-the-shelf machine learning algorithms to trade on real, live
financial markets.

Since `more data beats better
algorithms <https://www.quora.com/In-machine-learning-is-more-data-always-better-than-better-algorithms>`__,
Sairen is focused on intraday trading at the frequency of minutes or seconds
(not `HFT <https://en.wikipedia.org/wiki/High-frequency_trading>`__).
The environment provides observations in the form of real-time market data (quotes, bars, ticks) and your AI agent
issues actions in the form of orders to buy or sell.  The reward is profit or loss.  Oh, and Sairen *only* runs
live -- there is no backtesting and no market simulator, though you can (and REALLY SHOULD) use paper money.

.. |statusbadge| image:: https://img.shields.io/badge/status-alpha-red.svg
.. |buildbadge| image:: https://gitlab.com/doctorj/sairen/badges/master/build.svg
                :target: projlink_

|buildbadge| |statusbadge|

.. GitLab README doesn't render TOC (where would it link anyway) or sphinx roles, so we have to put in normal links.

* `GitLab Repo <https://gitlab.com/doctorj/sairen/>`__
* `Documentation <https://doctorj.gitlab.io/sairen/>`__:
  `Quickstart <https://doctorj.gitlab.io/sairen/quickstart.html>`__ |
  `Examples <https://doctorj.gitlab.io/sairen/examples.html>`__ |
  `Observations and Actions <https://doctorj.gitlab.io/sairen/observations.html>`__ |
  `Transforms <https://doctorj.gitlab.io/sairen/xforms.html>`__ |
  `API <https://doctorj.gitlab.io/sairen/api.html>`__


.. toctree::
   :maxdepth: 2
   :hidden:

   Home <self>
   quickstart
   observations
   xforms
   examples
   api


Is that a good idea?
--------------------

No, no it is not. But can you resist the allure of sitting back and sipping Mai Tais while your AI
learns to print money? No, no you cannot.

Hence the name.

How Bad Is It?
--------------

Well, trading a single futures contract, you can lose $12.50 a tick plus $4 in commissions + fees every
second without breaking a sweat. So you can easily **lose a thousand dollars a minute**, or $400,000 a day just working
banker's hours. I haven't even dipped into options. Your losses really are theoretically unlimited.

Did I mention this is a terrible, terrible idea?

Dear ``$DIETY``, WHY?
---------------------

When the `Singularity <https://en.wikipedia.org/wiki/Technological_singularity>`__ arrives, it's going to
need a way to fund itself.  Perhaps I will find favor in the eyes of our new Singular Overlord before
our species is summarily squashed.

No but seriously:

* More data
* Bigger models
* Quicker feedback
* Less overfitting
* Adaptive strategies
* More (opportunity for) profit

Trading on low-frequency, say, daily data, you have ~250 prices per year going back
perhaps tens of years. That's thousands of data points, not really enough to train
"serious" models without overfitting. (And is the price of IBM 20 years ago really meaningful today
anyway?) Should you manage to find a historically profitable strategy, it might take months to see
if it really works going forward. That's a slow development cycle. By contrast, there are
23,400 one-second bars in a regular market day, and if your algorithm chokes on them you'll find out pretty quickly!

With hundreds of thousands or millions of data points (days or weeks worth of second-resolution data) you can start to
train bigger, deeper models.  A few hours or days of walk-forward testing will tell you if they're working.

Not to mention that backtesting is
`notorious <https://www.quora.com/I-developed-an-algorithm-that-makes-13-per-day-Whats-next/answer/Justin-Medlin-1>`__
for green-lighting strategies that crash and burn in real trading.  Not only that, the rare strategy that does
work out-of-sample often peters out before long due to changing market conditions.

Finally, in theory every price change is an opportunity to profit -- if you can predict it. If the
price changes once a day you can only profit so much. If it changes once a second, you can profit so
much more. (In practice, commissions, spreads, slippage, liquidity, latency, and plain ol' intestinal
fortitude put a floor on how low you can go -- not everyone can stomach losing a thousand dollars a
minute.)

In sum, intraday trading provides more opportunities to profit from more data, and walk-forward
optimization provides less opportunities to fool yourself. If your algorithm sucks (and it probably will)
you'll find out quickly.

Or, maybe I just own a lot of `IBKR <http://www.nasdaq.com/symbol/ibkr/real-time>`__.


How Do I Get Started?
---------------------

First, ask youself if the allure of skimming a few pennies off the top of a corrupt system that will
overtake your life, blacken your soul, and bankrupt you in the process is the best use of your talents. No? Good choice; back to
`Hacker News <https://news.ycombinator.com/>`__. Yes? Get a second opinion from your mother. I'll
wait.

Still nothing better to do with your life? `Don't say I didn't warn you <https://doctorj.gitlab.io/sairen/quickstart.html>`__.


Show Me The Code
----------------

.. literalinclude:: ../examples/trading_monkey.py

Check out the `examples <https://doctorj.gitlab.io/sairen/examples.html>`__ in the documentation.


Isn't that a supervised learning problem?
-----------------------------------------
It certainly can be.  The manifest reason to treat algorithmic trading as a reinforcement learning problem would be that
your actions affect the market, but that's probably only an issue if you're Goldman Sachs.  Another plausible reason is
that the most profitable action might depend on your current market position, unrealized profit, and holding period, but that's easy to
hand-code around.  Really, it's that accurate market simulators are tricky to write well, and I'm lazy.  And I like the idea
of an adaptive strategy that learns to print money while you sleep.  Plus RL is just
`so hot right now <https://www.youtube.com/watch?v=FAxJECJJG6w>`__.


But really, no simulator?
-------------------------

Sairen only runs against a live Interactive Brokers connection, whether paper money (highly recommended) or real money
(HIGHLY UN-RECOMMENDED) -- there is no simulated environment.  That means there is no offline training or backtesting;
you must learn on-line, in real time.  However, you can save and replay past observation-action-reward data yourself if
you like.

Typical gym environments let you (re-)run as many environments as you want; Sairen is different.
First, Sairen is not deterministic -- you get real live different data every time. :meth:`reset()`-ing the environment
plops you into the current stream of market data, wherever that happens to be.

Second, there is only *one* envrionment per financial instrument, the actual market it trades on.  Running more than one
agent would just be `self-trading <https://ptg.fia.org/articles/what-self-trade-anyway>`__ in a zero-sum game (and
losing after commissions).  (It also makes it hard to attribute PNL -- who wins if Agent 1 opens a trade that Agent 2
closes?) I like to think of it this way: what if the only players in the market were *two* instances of your
agent?  They can't both win (but your broker sure would).

Third, timing matters.  Observations arrive at regular intervals on the real, external-world wall clock.
This implies 1) you cannot learn faster than real time and 2) you cannot act slower than real time.  If your agent
falls behind, you start issuing orders no longer relevant to current market conditions and learning from stale data. So pausing to retrain
for ten minutes when data arrives every second is probably not a good idea (though maybe in another thread, it could
be).

A system for training multiple models at once is high on the list of priorities.


Why Interactive Brokers?
------------------------

IB has the best combination of API, client libraries, low commissions, market access, advanced order types, and fast executions.
OK, actually it has a terrible API, opaque documentation, thin client libraries, and exorbitant commissions; but that's the state of
the industry.  (Incidentally, want to actually make some money on the stock market?  Start a developer-oriented brokerage with a
first-class REST API.)

Can I contribute?
-----------------
You bet: https://gitlab.com/doctorj/sairen/


I feel like you should have a big fat disclaimer here
-----------------------------------------------------

This project is |statusbadge| and almost certainly contains bugs that will cause you to lose money.  It should go
without saying that it comes with no warranty, none of this is investment advice, past performance will actively
misrepresent future results, data may come from a `random number generator <http://dilbert.com/strip/2016-04-01>`__, and
you should not entrust your brokerage account to artificial intelligence.  Yet.

