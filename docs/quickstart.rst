Quickstart
==========

.. currentmodule:: sairen

Download IB Trader Workstation
------------------------------
The Interactive Brokers API is a little weird in that you don't connect directly to their servers;
you connect to their locally running desktop app, and it proxies your requests.  This is strange,
but it has the advantages that your API code doesn't have to worry about authentication, and you
get a free whiz-bang daytrading GUI to watch the action and impress your friends.  I'll be honest --
it's a pretty crappy application, but we're stuck with it.

So, the first step is to download and install the latest TWS for your operating system.

* https://www.interactivebrokers.com/en/index.php?f=14099#tws-software

It's a Java app, and the installer installs its own JRE.

Sign in to TWS and Enable API Access
------------------------------------
You can log in to TWS with username ``edemo``, password ``demouser``.  This will get you completely fake
market data -- not delayed, not sampled, but crappy-pseudo-random-number-generator *fake*.

Next, you need to enable API access in the TWS configuration:

https://interactivebrokers.github.io/tws-api/initial_setup.html#gsc.tab=0

Since the demo does not save its settings, you will have to do this every time you start TWS --
which, by the way, you will have to do every day, as it automatically restarts a bit before midnight.
Did I mention crappy?

If you sign up for a paper (unfunded) account, I believe you can get real-time forex and bond data.  To
get anything else, you'll need to fund an account with $10,000.

While it's installing you may want to look over the `User Notes <https://gitlab.com/doctorj/ibroke/blob/master/docs/notes.md#user-notes>`__
section of these IB notes to orient yourself to the weird world of Interactive Brokers.


Install Sairen and Run an Example
---------------------------------

Sairen requires Python 3.4+.  You may want to first create and activate a Python virtual environment for Sairen:

.. code-block:: bash

    python -m venv python-sairen
    . python-sairen/bin/activate

Check out the code and install the dependencies:

.. code-block:: bash

    git clone https://gitlab.com/doctorj/sairen.git
    cd sairen
    pip install -r requirements.txt

Then you should be able to run the examples:

.. code-block:: bash

    PYTHONPATH=. examples/trading_monkey.py

This should print something like this:

.. code-block:: console

    Server Version: 76
    TWS Time at connection:20170116 15:21:22 PST
    2017-01-16 15:21:28 [INFO] sairen.env: Sairen 0.3.0 trading ('AAPL', 'STK', 'SMART', 'USD', None, 0.0, None) up to 1 contracts
    2017-01-16 15:21:28 [WARNING] sairen.env: ALERT: Unhalt
    time      step     pnl    gain  reward action     position         bid/ask     sizes         last   volume
    23:21:32     0    0.00    0.00    0.00   0.00    0@0.00     120.45/120.47     4x1     120.44@1       35030
    23:21:33     1    0.00    0.00    0.00   0.10    0@0.00     120.45/120.48     3x33    120.44@1       35030
    23:21:34     2    0.00    0.00    0.00   0.43    0@0.00     120.44/120.46     5x35    120.44@1       35030
    23:21:35     3    0.00    0.00    0.00   0.21    0@0.00     120.44/120.46     5x35    120.44@1       35030
    23:21:36     4    0.00    0.00    0.00   0.09    0@0.00     120.44/120.46     5x34    120.44@1       35030

Trades should be reflected in the TWS GUI.  You can safely stop it with CTRL-C.

Congratulations, you're ready to start printing money!

Instrument tuples
-----------------
Here are some examples of specifying the ``instrument`` you pass to :class:`MarketEnv`:

* US Stocks: ``'AAPL'``
* Futures: ``('ES', 'FUT', 'GLOBEX', 'USD', '20170317')``
* Forex: ``('EUR', 'CASH', 'IDEALPRO')``

The general form is a 7-tuple: ``(symbol, sec_type, exchange, currency, expiry, strike, opt_type)``
where you can elide any trailing values that are unneeded.  If you omit the expiry date, Sairen will
choose the instrument with the nearest expiry.

symbol:
    The ticker symbol, or base currency for forex
sec_type:
    The security type for the contract;  stock ``STK``, futures ``FUT``, forex ``CASH``, options ``OPT``
exchange:
    The exchange to trade the contract on.  Usually, stock ``SMART``, futures ``GLOBEX``, forex ``IDEALPRO``
currency:
    The currency in which to purchase the contract (quote currency for forex), usually ``USD``
expiry:
    Future or option expiry date, format ``YYYYMMDD``.
strike:
    The strike price for options
opt_type:
    ``PUT`` or ``CALL`` for options


The best way to find instrument details is in TWS itself.  Right click an empty spot in Favorites Monitor, type in
the search term, choose the instrument you want.  When you have the instrument in Favorites, right-click and choose
Contract Info > Description.

You can also try IB's online `contract search tool <https://pennies.interactivebrokers.com/cstools/contract_info/v3.9/>`__, but
it leaves a lot to be desired.


Keeping up with real time
-------------------------
Coming soon.
