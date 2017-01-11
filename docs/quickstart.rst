Quickstart
==========

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
    TWS Time at connection:20170104 23:40:55 PST
    [2017-01-04 23:41:00,953] Sairen 0.1.0 trading ('ES', 'FUT', 'GLOBEX', 'USD', '20170317', 0.0, None) up to 1 contracts
    2017-01-05 07:41:03   0   PNL    0.00   unreal    0.00   rew    0.00   act  0.00   pos  0@0.00     2260.75/2260.50 2261x2261     nan@nan          0  0  1  1  0
    2017-01-05 07:41:04   1   PNL    0.00   unreal    0.00   rew    0.00   act  0.10   pos  0@0.00     2260.75/2260.50 2261x2261     nan@nan          0  0  1  1  0
    2017-01-05 07:41:05   2   PNL    0.00   unreal    0.00   rew    0.00   act  0.43   pos  0@0.00     2260.75/2260.50 2261x2261     nan@nan          0  0  1  1  0
    2017-01-05 07:41:06   3   PNL    0.00   unreal    0.00   rew    0.00   act  0.21   pos  0@0.00     2260.75/2260.50 2261x2261     nan@nan          0  0  1  1  0

Trades should be reflected in the TWS GUI.  You can safely stop it with CTRL-C.

Congratulations, you're ready to start printing money!

Contract tuples
---------------
Coming soon.

Keeping up with real time
-------------------------
Coming soon.
