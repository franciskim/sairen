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
from . import Bar, create_logger

LOG = create_logger('xform', logging.INFO)
NYSE_OPEN_TIME = datetime.time(14, 30)      # EST in UTC
NYSE_CLOSE_TIME = datetime.time(21, 00)


# time, bid, bidsize, ask, asksize, last, lastsize, lasttime, open, high, low, close, vwap, volume, open_interest


class Delta:
    """The difference between the last `lookback` bid, ask, last, and volume from `lookback + 1` bars ago."""
    def __init__(self, lookback):
        self.lookback = lookback
        self.vals = deque(maxlen=lookback + 1)
        obs_size = lookback * 4 + 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,))

    def __call__(self, bar, unrealized_gain, position_actual, max_quantity):
        bar = Bar._make(bar)
        self.vals.append([bar.bid, bar.ask, bar.last, bar.volume / 100])
        obs = np.hstack([np.diff(np.array(self.vals), axis=0).flatten(), unrealized_gain / 100, position_actual])
        if len(self.vals) == self.lookback + 1 and np.isfinite(obs).all():
            assert self.observation_space.contains(obs), 'shape {} expected {}\n{}'.format(obs.shape, self.observation_space.shape, obs)
        else:
            obs = None
        LOG.debug('XFORM %s', obs)
        return obs


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

    def __call__(self, bar, unrealized_gain, position_actual, max_quantity):
        assert len(bar) == len(Bar._fields)
        bar = Bar._make(bar)
        dt = datetime.datetime.utcfromtimestamp(bar.time)

        change = False
        if not math.isnan(bar.bid) and bar.bid != (self.bids[-1] if self.bids else None):
            self.bids.append(bar.bid)
            change = True
        if not math.isnan(bar.ask) and bar.ask != (self.asks[-1] if self.asks else None):
            self.asks.append(bar.ask)
            change = True
        if not math.isnan(bar.vwap) and bar.vwap:      # Average trade price is closer to bid than ask
            self.atbid.append(abs(bar.vwap - bar.bid) < abs(bar.vwap - bar.ask))
            change = True

        unrealized_gain_sign = np.sign(unrealized_gain)      # [-1, 1] keeps it simple
        afterhours = 0 if (0 <= dt.weekday() <= 4 and NYSE_OPEN_TIME <= dt.time() <= NYSE_CLOSE_TIME) else 1    # TODO: Put actual hours on Instrument and use those
        bid_change = tuple(self.bids[i] > self.bids[i - 1] for i in range(1, len(self.bids)))      # 0 for downtick, 1 for uptick
        last_at_bid = tuple(self.atbid)     # 0 if last trades were closer to bid, 1 if closer to ask
        rel_position = np.clip(position_actual / max_quantity, -1.0, 1.0)   # Scale and clip to [-1, 1]

        if not change or len(bid_change) < self.lookback or len(last_at_bid) < self.lookback:
            if len(self.bids) < self.bids.maxlen:
                LOG.debug('Not enough action: %d bid changes, %d trades', len(bid_change), len(last_at_bid))
            return None

        obs = np.asarray((rel_position, unrealized_gain_sign, afterhours) + bid_change + last_at_bid)
        return obs
