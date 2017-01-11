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

import numpy as np
from gym.spaces import Box
from ibroke import Quote

NYSE_OPEN_TIME = datetime.time(14, 30)      # EST in UTC
NYSE_CLOSE_TIME = datetime.time(21, 00)


class BinaryDelta:
    """Takes raw quotes and returns the sign of the changes in bid and whether the last trade
    was closer to the bid or ask."""
    def __init__(self, lookback):
        """:param int lookback: Number of previous quotes to compute deltas for."""
        self.lookback = lookback
        self.bids = deque(maxlen=lookback + 1)
        self.asks = deque(maxlen=lookback + 1)
        self.atbid = deque(maxlen=lookback)        # No +1 cause not computing deltas
        #: pos x unrealized_gain x afterhours x bid_changes x last_at_bids
        self.observation_space = Box(np.array([-1, -1, 0] + [0] * 2 * self.lookback), np.array([1, 1, 1] + [1] * 2 * self.lookback))

    # TODO: Need quotes or maybe account update messages to compute unrealized PNL

    def __call__(self, quote, unrealized_gain, position_actual, max_quantity, logger):
        logger.debug('XFORM %s', quote)
        assert len(quote) == len(Quote._fields)
        quote = Quote._make(quote)
        dt = datetime.datetime.utcfromtimestamp(quote.time)

        change = False
        if not math.isnan(quote.bid) and quote.bid != (self.bids[-1] if self.bids else None):
            self.bids.append(quote.bid)
            change = True
        if not math.isnan(quote.ask) and quote.ask != (self.asks[-1] if self.asks else None):
            self.asks.append(quote.ask)
            change = True
        if not math.isnan(quote.vwap) and quote.vwap:      # Average trade price is closer to bid than ask
            self.atbid.append(quote.vwap <= (quote.ask - quote.bid) / 2)
            change = True

        unrealized_gain_sign = np.sign(unrealized_gain)      # [-1, 1] keeps it simple
        afterhours = 0 if (0 <= dt.weekday() <= 4 and NYSE_OPEN_TIME <= dt.time() <= NYSE_CLOSE_TIME) else 1    # TODO: Put actual hours on Instrument and use those
        bid_change = tuple(self.bids[i] > self.bids[i - 1] for i in range(1, len(self.bids)))      # 0 for downtick, 1 for uptick
        last_at_bid = tuple(self.atbid)     # 0 if last trades were closer to bid, 1 if closer to ask
        rel_position = np.clip(position_actual / max_quantity, -1.0, 1.0)   # Scale and clip to [-1, 1]

        if not change or len(bid_change) < self.lookback or len(last_at_bid) < self.lookback:
            if len(self.bids) < self.bids.maxlen:
                logger.debug('Not enough action: %d bid changes, %d trades', len(bid_change), len(last_at_bid))
            return None

        obs = np.asarray((rel_position, unrealized_gain_sign, afterhours) + bid_change + last_at_bid)
        logger.debug('XFORM RET %s', obs)
        return obs
