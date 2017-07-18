"""
Sairen market data observation transformations.
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

import datetime
import math
from collections import deque
import logging

import numpy as np
from gym.spaces import Box
from . import Obs, create_logger

LOG = create_logger('xform', logging.INFO)
NYSE_OPEN_TIME = datetime.time(13, 30)      # EDT in UTC
NYSE_CLOSE_TIME = datetime.time(20, 00)


# time, bid, bidsize, ask, asksize, last, lastsize, lasttime, open, high, low, close, vwap, volume, open_interest, position, unrealized_gain


class Delta:
    """The difference between the last `lookback` bid, ask, last, and volume from `lookback + 1` bars ago."""
    def __init__(self, lookback):
        self.lookback = lookback
        self.vals = deque(maxlen=lookback + 1)
        obs_size = lookback * 4 + 2     # diffs for bid/ask/last/vol + unrealized/pos
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,))

    def __call__(self, obs):
        obs = Obs._make(obs)
        self.vals.append([obs.bid, obs.ask, obs.last, obs.volume / 100])
        xobs = np.hstack([np.diff(np.array(self.vals), axis=0).flatten(), obs.unrealized_gain / 100, obs.position])
        if len(self.vals) == self.lookback + 1 and np.isfinite(xobs).all():
            assert self.observation_space.contains(xobs), 'shape {} expected {}\n{}'.format(xobs.shape, self.observation_space.shape, xobs)
        else:
            xobs = None
        LOG.debug('XFORM %s', xobs)
        return xobs


class CashDelta:
    """The difference between the last `lookback` bid, ask and sizes from `lookback + 1` bars ago.
    For cash (forex) instruments without last or volume info."""
    def __init__(self, lookback):
        self.lookback = lookback
        self.vals = deque(maxlen=lookback + 1)
        obs_size = lookback * 4 + 2     # diffs for bid, bidsize, ask, asksize
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,))

    def __call__(self, obs):
        obs = Obs._make(obs)
        self.vals.append([obs.bid, obs.bidsize, obs.ask, obs.asksize])
        xobs = np.hstack([np.diff(np.array(self.vals), axis=0).flatten(), obs.unrealized_gain, obs.position])
        if len(self.vals) == self.lookback + 1 and np.isfinite(xobs).all():
            assert self.observation_space.contains(xobs), 'shape {} expected {}\n{}'.format(xobs.shape, self.observation_space.shape, xobs)
        else:
            xobs = None
        LOG.debug('XFORM %s', xobs)
        return xobs


class BinaryDelta:
    """Takes raw quotes and returns the sign of the changes in bid and whether the last trade
    was closer to the bid or ask."""
    def __init__(self, lookback):
        """:param int lookback: Number of previous quotes to compute deltas for."""
        self.lookback = lookback
        self.bids = deque(maxlen=lookback + 1)
        self.asks = deque(maxlen=lookback + 1)
        self.atbid = deque(maxlen=lookback)        # No +1 because not computing deltas
        #: pos x unrealized_gain x afterhours x bid_changes x last_at_bids
        self.observation_space = Box(np.array([-1, -1, 0] + [0] * 2 * self.lookback), np.array([1, 1, 1] + [1] * 2 * self.lookback))

    def __call__(self, obs):
        assert len(obs) == len(Obs._fields)
        obs = Obs._make(obs)
        dt = datetime.datetime.utcfromtimestamp(obs.time)

        change = False
        if not math.isnan(obs.bid) and obs.bid != (self.bids[-1] if self.bids else None):
            self.bids.append(obs.bid)
            change = True
        if not math.isnan(obs.ask) and obs.ask != (self.asks[-1] if self.asks else None):
            self.asks.append(obs.ask)
            change = True
        if not math.isnan(obs.vwap) and obs.vwap:      # Average trade price is closer to bid than ask
            self.atbid.append(abs(obs.vwap - obs.bid) < abs(obs.vwap - obs.ask))
            change = True

        unrealized_gain_sign = np.sign(obs.unrealized_gain)      # [-1, 1] keeps it simple
        afterhours = 0 if (0 <= dt.weekday() <= 4 and NYSE_OPEN_TIME <= dt.time() <= NYSE_CLOSE_TIME) else 1    # TODO: Put actual hours on Instrument and use those
        bid_change = tuple(self.bids[i] > self.bids[i - 1] for i in range(1, len(self.bids)))      # 0 for downtick, 1 for uptick
        last_at_bid = tuple(self.atbid)     # 0 if last trades were closer to bid, 1 if closer to ask
        rel_position = np.clip(obs.position, -1.0, 1.0)

        if not change or len(bid_change) < self.lookback or len(last_at_bid) < self.lookback:
            if len(self.bids) < self.bids.maxlen:
                LOG.debug('Not enough action: %d bid changes, %d trades', len(bid_change), len(last_at_bid))
            return None

        xobs = np.asarray((rel_position, unrealized_gain_sign, afterhours) + bid_change + last_at_bid)
        return xobs


class Basic:
    """Mean, stddev, min, max, for the last `lookback` bars, deltas for the last `deltas` bars, and most
    recent actual values of `fields`."""
    def __init__(self, lookback, deltas=4, fields=('bid', 'bidsize', 'ask', 'asksize')):
        self.lookback = int(lookback)
        self.fields = [Obs._fields.index(field) for field in fields]        # Can't be tuple for ndarray slicing to work
        self.vals = deque(maxlen=lookback)
        self.deltas = int(deltas)
        assert self.lookback > self.deltas >= 0
        #obs_size = len(fields) * (5 + deltas)       # mean, std, min, max, last, [deltas] for each field
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(5 + deltas, len(fields)))
        #       bid ask bidsize asksize
        # mean
        # std
        # min
        # max
        # last
        # delta1
        # ...

    def __call__(self, obs):
        obs = np.asarray(obs)
        assert obs.shape == (len(Obs._fields),)
        self.vals.append(obs[self.fields])
        vals = np.array(self.vals)
        xobs = np.vstack([vals.mean(axis=0), vals.std(axis=0), vals.min(axis=0), vals.max(axis=0), vals[-1], np.diff(vals, axis=0)[-self.deltas:]])
        if len(self.vals) > self.deltas and np.all(np.isfinite(xobs)):
            assert self.observation_space.contains(xobs), 'shape {} expected {}\n{}'.format(xobs.shape, self.observation_space.shape, xobs)
        else:
            xobs = None
        LOG.debug('XFORM %s', xobs)
        return xobs
