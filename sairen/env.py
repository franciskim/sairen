"""
Sairen environments.
"""

# Copyright (C) 2016  Doctor J
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from datetime import datetime, timedelta
import logging
import threading
import time
from queue import Queue
from collections import deque, namedtuple
from typing import Any, Tuple, Dict

import gym
import numpy as np
import sys
from io import StringIO
from gym.envs.registration import EnvSpec
from gym.spaces import Box
from gym.utils import EzPickle
from ibroke import IBroke, Bar, create_logger, Instrument, now

__version__ = "0.4.1"
__all__ = ('MarketEnv', 'Obs')
RENDER_HEADERS_EVERY_STEPS = 60     #: Print column names to stdout for human-rendered output every this many steps
# These are used to bound observation Boxes, not sure how important it really is.
MAX_INSTRUMENT_PRICE = 1e6
MAX_INSTRUMENT_VOLUME = 1e9
MAX_INSTRUMENT_QUANTITY = 20000
MAX_TRADE_SIZE = 1e6
MAX_TIME = time.time() + 10 * 365 * 24 * 60 * 60
Obs = namedtuple('Obs', Bar._fields + ('position', 'unrealized_gain'))      # type: ignore
Obs.__doc__ = Bar.__doc__.replace('Bar', 'Observation') + """
position
    A float usually in [-1, 1] giving your current actually held position as a fraction of `max_quantity` (negative for short).
    The integer number of shares held is ``int(position * max_quantity)``.

unrealized_gain
    The amount you would make if you liquidated your current position (bought at the ask or sold at the bid).  Negative for loss.
"""     # That's my favorite dirty hack in a while :)
OBS_BOUNDS = Obs(time=MAX_TIME, bid=MAX_INSTRUMENT_PRICE, bidsize=MAX_TRADE_SIZE, ask=MAX_INSTRUMENT_PRICE, asksize=MAX_TRADE_SIZE, last=MAX_INSTRUMENT_PRICE, lastsize=MAX_TRADE_SIZE, lasttime=MAX_TIME, open=MAX_INSTRUMENT_PRICE, high=MAX_INSTRUMENT_PRICE, low=MAX_INSTRUMENT_PRICE, close=MAX_INSTRUMENT_PRICE, vwap=MAX_INSTRUMENT_PRICE, volume=MAX_INSTRUMENT_VOLUME, open_interest=MAX_INSTRUMENT_VOLUME, position=1, unrealized_gain=MAX_INSTRUMENT_PRICE)      # type: ignore


class MarketEnv(gym.Env, EzPickle):
    """Access the Interactive Brokers trading API as an OpenAI Gym environment.

    ``MarketEnv`` provides :doc:`observations <observations>` of real-time market data for a single financial instrument.
    The action is a float in the range [-1, 1] to set the (absolute) target position in that instrument.

    Calling :meth:`close()` (or terminating Python) will cancel any open orders, flatten positions, and
    disconnect from IB.
    """

    action_space = Box(-1, 1, shape=(1,))
    """
    MarketEnv's action is a continuous float from -1 to 1 that sets the target position as a fraction of the
    environment's ``max_quantity`` parameter.  -1 means set the position to short ``max_quantity``, 0 means
    exit/close/flatten/no position, and 1 means set the position to long ``max_quantity``. These are "target" positions,
    so an action of 1 means "regardless of current position, buy or sell (or do nothing) as necessary to make my position
    ``max_quantity``."  Intermediate values are scaled by ``max_quantity`` and rounded to the nearest multiple of 
    ``quantity_increment``.  Orders are issued at market price so they are filled quickly.
    """
    metadata = {'render.modes': ['human', 'ansi']}

    # IBroke is event-driven (its methods are called asynchronously by IBPy), whereas Env is essentially an external
    # iterator (the caller calls step() when it's ready for the next observation). To play together, when IBroke's on_bar()
    # callback receives a new bar (observation), it's stored in a queue.  When Env.step() is called, it takes the next observation
    # out.  If no observations are available, step() will block waiting for an observation to appear in the queue. If more than one
    # observation is available, it means that step() is falling behind on processing observations, and a warning will be printed.

    def __init__(self, instrument, max_quantity=1, quantity_increment=1, obs_type='time', obs_size=1, obs_xform=None, episode_steps=None, host='localhost', port=7497, client_id=None, timeout_sec=5, afterhours=True, loglevel=logging.INFO):
        """
        :param str,tuple instrument: ticker string or :class:`IBroke` ``(symbol, sec_type, exchange, currency, expiry, strike, opt_type)`` tuple.
        :param int max_quantity: The number of shares/contracts that will be bought (or sold) when the action is 1 (or -1).
        :param int quantity_increment: The minimum increment in which shares/contracts will be bought (or sold).  The actual number for a given
          action is ``round(action * max_quantity / quantity_increment) * quantity_increment``, clipped to the range ``[-max_quantity, max_quantity]``.
        :param str obs_type: ``time`` for bars at regular intervals, or ``tick`` for bars at every quote change.
          Raw observations are numpy float ndarrays with the following fields::

                time, bid, bidsize, ask, asksize, last, lastsize, lasttime,
                open, high, low, close, vwap, volume, open_interest, position, unrealized_gain

          See the :class:`Obs` convenience namedtuple for detailed field descriptions.
        :param float obs_size: How often you get an observation in seconds.  Ignored for ``obs_type='tick'``.
        :param func obs_xform: Callable that takes a raw input observation array and transforms it,
          returning either another numpy array or ``None`` to indicate data is not ready yet.
        :param int,None episode_steps: Number of steps after ``reset()`` to run before returning `done`, or ``None`` to run indefinitely.
          The final step in an episode will have its action forced to close any open positions so PNL can be properly accounted.
        :param int client_id: A unique integer identifying which API client made an order.  Different instances of Sairen running at the same time must use
          different ``client_id`` values.  In order to discover and modify pre-existing open orders, you must use the same ``client_id`` the orders were created with.
        :param timeout_sec: request timeout in seconds used by IBroke library.
        :param afterhours: If True, operate during normal market and after hours trading; if False, only operate during normal market hours.
        :param int loglevel: The `logging level <https://docs.python.org/3/library/logging.html#logging-levels>`_ to use.
        """
        gym.Env.__init__(self)      # EzPickle is supposed to (un)pickle this object by saving the args and creating a new one with them.  Otherwise the IBroke and maybe the queues aren't serializable.
        EzPickle.__init__(self, instrument=instrument, max_quantity=max_quantity, min_quantity=quantity_increment, obs_type=obs_type, obs_size=obs_size, obs_xform=obs_xform, episode_steps=episode_steps, host=host, port=port, client_id=client_id, timeout_sec=timeout_sec, afterhours=afterhours, loglevel=loglevel)
        self.log = create_logger('sairen', loglevel)
        self.max_quantity = int(max_quantity)
        self.quantity_increment = int(quantity_increment)
        assert 1 <= self.quantity_increment <= self.max_quantity and self.max_quantity <= MAX_INSTRUMENT_QUANTITY, (self.quantity_increment, self.max_quantity)
        self.episode_steps = None if episode_steps is None else int(episode_steps)
        assert self.episode_steps is None or self.episode_steps > 0
        self.afterhours = afterhours
        self.obs_type = obs_type
        self.data_q = None      # Initialized in _reset
        self.profit = 0.0       # Since last step; zeroed every step
        self.episode_profit = 0.0     # Since last reset
        self.reward = None      # Save most recent reward so we can use it in render()
        self.raw_obs = None     # Raw obs as ndarray
        self.observation = None # Most recent transformed observation
        self.pos_desired = 0    # Action translated into target number of contracts
        self.done = True        # Start in the "please call reset()" state
        self.step_num = 0       # Count calls to step() since last reset()
        self.unrealized_gain = 0.0
        self._finish_on_next_step = False
        assert obs_xform is None or callable(obs_xform)
        self._xform = (lambda obs: obs) if obs_xform is None else obs_xform         # Default xform is identity

        self.ib = IBroke(host=host, port=port, client_id=client_id, timeout_sec=timeout_sec, verbose=2)
        self.instrument = self.ib.get_instrument(instrument)
        self.log.info('Sairen %s trading %s up to %d contracts', __version__, self.instrument.tuple(), self.max_quantity)
        market_open = self.market_open()        #self.ib.market_open(self.instrument, afterhours=self.afterhours)
        self.log.info('Market {} ({} hours).  Next {} {}'.format('open' if market_open else 'closed', 'after' if self.afterhours else 'regular', 'close' if market_open else 'open', self.ib.market_hours(self.instrument, self.afterhours)[int(market_open)]))
        self.ib.register(self.instrument, on_bar=self._on_mktdata, bar_type=obs_type, bar_size=obs_size, on_order=self._on_order, on_alert=self._on_alert)
        self.observation_space = getattr(obs_xform, 'observation_space', Box(low=np.zeros(len(OBS_BOUNDS)), high=np.array(OBS_BOUNDS)))     # TODO: Some bounds (pos, gain) are negative
        self.log.debug('XFORM %s', self._xform)
        self.log.debug('OBS SPACE %s', self.observation_space)
        np.set_printoptions(linewidth=9999)
        self.pos_actual = self.ib.get_position(self.instrument)     # Actual last reported number of contracts held
        self.act_start_time = None
        self.act_time = deque(maxlen=10)        # Track recent agent action times
        self.spec = EnvSpec('MarketEnv-{}-v0'.format('-'.join(map(str, self.instrument.tuple()))), trials=10, max_episode_steps=episode_steps, nondeterministic=True)      # This is a bit of a hack for rllab

    def _on_mktdata(self, instrument: Instrument, bar: Bar) -> None:
        """Called by IBroke on new market data; transforms observation and, if ready, puts it in data_q.

        After a :meth:`reset`, transforms are being called (updated) in the background even when :meth:`step` is
        not called.
        """
        self.pos_actual = self.ib.get_position(self.instrument)
        self.unrealized_gain = self.pos_actual * self.instrument.leverage * ((bar.bid if self.pos_actual > 0 else bar.ask) - (self.ib.get_cost(self.instrument) or 0))     # If pos > 0, what could we sell for?  Assume buy at the ask, sell at the bid
        self.raw_obs = np.array(bar + (self.pos_actual / self.max_quantity, self.unrealized_gain), dtype=float)
        self.log.debug('OBS RAW %s', self.raw_obs)
        obs = self._xform(self.raw_obs)
        self.log.debug('OBS XFORM %s', obs)
        assert obs is None or isinstance(obs, np.ndarray)

        if obs is not None and self.ib.connected and not self.done and self.data_q is not None:     # guard against step() being called before reset().  It also turns out that you can still receive market data while "disconnected"...
            self.data_q.put_nowait(obs)
            if self.data_q.qsize() > 1:
                self.log.warning('Your agent is falling behind! Observation queue contains %d items.', self.data_q.qsize())

    def _on_order(self, order) -> None:
        """Called when order status changes by IBroke."""
        self.log.debug('ORDER %s\t(thread %d)', order, threading.get_ident())
        self.profit += order.profit

    def _on_alert(self, instrument, msg) -> None:
        self.log.warning('ALERT: %s', msg)

    def flatten(self) -> None:
        """Cancel any open orders and close any positions."""
        if hasattr(self, 'instrument'):     # If self.ib times out connecting, we don't want to flatten() atexit.
            self.ib.flatten(self.instrument)
            time.sleep(1)       # Give order time to fill  TODO: Wait (with timeout) for actual fill

    def finish_on_next_step(self) -> None:
        """Sets a flag so that the next call to :meth:`step` will flatten any positions and return ``done = True``."""
        self._finish_on_next_step = True

    def market_open(self) -> bool:
        """:Return: True if the market will be open in the very near future (respecting the value of `afterhours`)."""
        return self.ib.market_open(self.instrument, now() + timedelta(seconds=self.ib.timeout_sec), afterhours=self.afterhours)

    @property
    def info(self) -> Dict[str, Any]:
        """A dict of information useful for monitoring the environment."""
        return {
            'step': self.step_num,
            'episode_profit': self.episode_profit,
            'position_desired': self.pos_desired,
            'position_actual': self.ib.get_position(self.instrument),
            'unrealized_gain': self.unrealized_gain,
            'avg_cost': self.ib.get_cost(self.instrument) or 0.0,
            'agent_time_last': self.act_time[-1] if self.act_time else np.nan,
            'agent_time_avg': np.mean(self.act_time) if self.act_time else np.nan,
        }

    def _close(self) -> None:
        """Cancel open orders, flatten position, and disconnect."""
        self.log.info('Cancelling, closing, disconnecting.')
        if hasattr(self, 'ib'):     # We may not have ever connected, but _close gets called atexit anyway.
            self.done = True        # Stop observations going into the queue
            self.flatten()
            self.ib.disconnect()

    def _reset(self) -> np.ndarray:
        """Flatten positions, reset accounting, and return the first observation.

        Discards any existing observations in the queue.
        """
        self.log.debug('RESET')
        # TODO: Warn if position is not zero at start of episode
        self.done = True        # Prevent _on_mktdata() from putting things in the queue and triggering step() while we flatten
        self.flatten()
        self.profit = 0.0
        self.episode_profit = 0.0
        self.reward = 0.0
        self.unrealized_gain = 0.0
        self.observation = None
        # Note: even when we're "disconnected" or the market is "closed," we can still receive market data (separate connection, afterhours can be open).
        while not self.ib.connected:
            time.sleep(self.ib.timeout_sec)
        msg = None
        while not self.market_open():
            if msg is None:
                open_, _ = self.ib.market_hours(self.instrument, afterhours=self.afterhours)
                msg = 'Market is closed.'
                if open_:
                    msg += ' Next open is {} mins ({})'.format(int(np.ceil((open_ - now()).total_seconds() / 60)), open_)
                self.log.info(msg)
            time.sleep(self.ib.timeout_sec)
        self.done = False
        self.action = 0.0
        self.pos_desired = 0
        self._finish_on_next_step = False
        self.step_num = 0
        self.data_q = Queue()
        self.observation = self.data_q.get()       # Blocks until obs ready
        self.act_start_time = time.time()
        return self.observation

    def _step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute any trades necessary to allocate the position to the float `action` in [-1, 1] (short, long),
        wait for the next (transformed) observation, and return the observation and reward.

        Will return ``done = True`` if ``self.episode_steps`` has been reached, the connection has been lost, just before the market closes,
        or :meth:`finish_on_next_step` has been called.  When done, the final observation will be all zeros.
        """
        self.act_time.append(time.time() - self.act_start_time)
        self.step_num += 1
        self.log.debug('STEP {}: {}\t({:.2f}s)'.format(self.step_num, action, self.act_time[-1]))
        if self.done:
            raise ValueError("I'm done, yo.  Call reset() if you want another play.")

        # If last step, set action to flatten, done = True
        done = False
        if self._finish_on_next_step or (self.episode_steps is not None and self.step_num >= self.episode_steps) or not self.ib.connected or not self.market_open():
            if not self.market_open():
                self.log.info('Market closing.')
            action = 0.0
            done = True     # Don't set self.done before waiting on self.data_q, because it will never put anything in.

        self.action = float(action)        # Save raw un-clipped action (but make sure it's a float)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), 'action {}, low {}, high {}'.format(action, self.action_space.low, self.action_space.high)       # requires an array
        action = np.asscalar(action)

        # Issue order to take action
        self.ib.cancel_all(self.instrument)
        position = self.ib.get_position(self.instrument)
        open_orders = sum(1 for _ in self.ib.get_open_orders())
        self.pos_desired = int(np.clip(round(action * self.max_quantity / self.quantity_increment) * self.quantity_increment, -self.max_quantity, self.max_quantity))
        # Try to prevent orders and/or positions piling up when things get busy.
        if open_orders > 1 or (abs(position) > self.max_quantity and abs(self.pos_desired) >= abs(position)):
            self.log.warning('Constipation: position %d, %d open orders, skipping action.', position, open_orders)
        else:
            self.log.debug('ORDER TARGET %d', self.pos_desired)
            self.ib.order_target(self.instrument, self.pos_desired)

        if done:
            # TODO: Actually wait until order settles.  (Close is not happening or accounting is not good.)
            time.sleep(1)       # Wait for final close order to fill

        self.reward = self.profit       # Reward is profit since last step
        self.episode_profit += self.profit
        self.profit = 0
        if done:
            self.observation = np.zeros(self.observation_space.shape)
        else:
            self.observation = self.data_q.get()       # block until next obs ready

        self.done = done        # Don't set until after waiting on queue, or queue will never get filled.
        info = self.info        # Variable because computed property, used more than once, want to be consistent.
        self.log.debug('OBS %s\tINFO %s', self.observation, info)
        self.log.debug('REWARD %.2f\tDONE %s', self.reward, self.done)
        self.act_start_time = time.time()
        return self.observation, self.reward, self.done, info

    def _render(self, mode='human', close=False):
        if mode in ('human', 'ansi'):
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            if not close:
                if self.instrument.sec_type == 'CASH':
                    FIELDS = (
                        ('time', '{time}', 8, 'UTC observation timestamp'),
                        ('step', '{step:d}', '>5', 'Step number in this episode (first action is step 1)'),
                        ('pnl', '{pnl:.2f}', '>7', 'Episode profit'),
                        ('unreal', '{unreal:.2f}', '>7', 'Episode unrealized gain'),
                        ('reward', '{reward:.2f}', '>7', 'Last reward'),
                        ('action', '{action: 6.2f}', '>6', 'Last action (raw float)'),
                        ('position', '{pos: 6d}@{cost:<7.5f}', '>17', 'Actual shares/contracts currently held'),
                        ('bid/ask', '{bid:8.5f}/{ask:<8.5f}', '>18', 'Most recent bid and ask prices'),
                        ('sizes(k)', '{bidsize:5.0f}x{asksize:<5.0f}', '>11', 'Most recent bid and ask sizes (in thousands)'))
                else:
                    FIELDS = (
                        ('time', '{time}', 8, ''),
                        ('step', '{step:d}', '>5', ''),
                        ('pnl', '{pnl:.2f}', '>7', ''),
                        ('unreal', '{unreal:.2f}', '>7', ''),
                        ('reward', '{reward:.2f}', '>7', ''),
                        ('action', '{action: 6.2f}', '>6', ''),
                        ('position', '{pos: 4d}@{cost:<7.2f}', '>12', ''),
                        ('bid/ask', '{bid:7.2f}/{ask:<7.2f}', '>15', ''),
                        ('sizes', '{bidsize:4.0f}x{asksize:<4.0f}', '>9', 'Most recent bid and ask sizes'),
                        ('last', '{lastsize: 4.0f}@{last:<7.2f}', '>12', 'Most recent trade price'),
                        ('volume', '{volume:>8.0f}', '>8', 'Total cumulative volume for the day'))

                if self.step_num % RENDER_HEADERS_EVERY_STEPS == 1:
                    print(*('{:{}}'.format(name, width) for name, _, width, _ in FIELDS), file=outfile)
                data = dict(dict(zip(Obs._fields, self.raw_obs)), step=self.step_num, reward=self.reward, unreal=self.unrealized_gain, action=self.action, pnl=self.episode_profit, pos=int(self.pos_actual), cost=self.info['avg_cost'] or 0.0, raw_obs=self.raw_obs, time=datetime.utcfromtimestamp(round(self.raw_obs[0])).time())
                if self.instrument.sec_type == 'CASH':      # Sizes in thousands, since minimum is 20,000.
                    data['bidsize'] //= 1000
                    data['asksize'] //= 1000
                print(*('{:{}}'.format(fmt.format(**data), width) for _, fmt, width, _ in FIELDS), file=outfile)
                self.log.debug('INFO %s', sorted(self.info.items()))
                if mode == 'ansi':
                    return outfile
        else:
            raise NotImplementedError("Render mode '{}' not implemented".format(mode))

    def _seed(self, seed=None):
        raise Warning("Don't you wish you could seed() the stock market!")
