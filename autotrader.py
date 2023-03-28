# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import statistics

from typing import List

from scipy import stats

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 10
POSITION_LIMIT = 93
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


class AutoTrader(BaseAutoTrader):
    """Auto-trader.

    Record the average, volume-weighted bid/ask, and use the average of nearest 6 as ask/ bid prices. 
    Calculate the rate of change of bid/ ask prices to adjust the average price to follow the market's momentum.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        #Store the last 10 average price for momentum
        self.history_vbid = []
        self.history_vask = []
        self.history_mid = []
        self.gain = -1 #Store average gain for RSI
        self.loss = -1 #Store average loss for RSI
        self.bid_ordered = 0
        self.ask_ordered = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info("received order book for instrument %d with sequence number %d", instrument,
                         sequence_number)
        if instrument == Instrument.FUTURE:
            if bid_prices[0] != 0:
                vbid = 0
                vask = 0
                for i in range(1, 4):
                    vbid += bid_prices[i] * bid_volumes[i]
                    vask += ask_prices[i] * ask_volumes[i]
                vbid = (vbid // sum(bid_volumes[1:4])) if bid_prices[0] != 0 else 0
                vask = (vask // sum(ask_volumes[1:4])) if bid_prices[0] != 0 else 0
                vmid = (vbid + vask) // 2

                
                
                self.history_vbid.append(vbid)
                self.history_vask.append(vask)
                self.history_mid.append(vmid)



                over_bought_rsi = False
                over_sold_rsi = False
                rsi = -1

                if len(self.history_vbid) > 15:
                    self.history_vbid = self.history_vbid[-15:]
                    self.history_vask = self.history_vask[-15:]
                    self.history_mid = self.history_mid[-15:]
                
                if len(self.history_mid) > 10:
                    if self.gain == -1:
                        gtotal = 0
                        ltotal = 0
                        for i in range(-11, -1):
                            diff = vmid - self.history_mid[i]
                            if diff > 0:
                                gtotal += diff
                            else:
                                ltotal += abs(diff)
                        self.gain = gtotal / 10
                        self.loss = ltotal / 10
                    else:
                        gain = 0
                        loss = 0
                        diff = vmid - self.history_mid[-2]
                        if diff > 0:
                            gain = diff
                        else:
                            loss = abs(diff)
                        self.gain = (self.gain * 9 + gain)/10
                        self.loss = (self.loss * 9 + loss)/10
                    rs = self.gain/ self.loss
                    rsi = 100 - 100/(1+rs)
                    if rsi > 75:
                        over_bought_rsi = True
                    elif rsi < 25:
                        over_sold_rsi = True
                                       
                pre_end = len(self.history_vask) - 1
                avg_vbid = 0
                avg_vask = 0

                for i in range(max(0, pre_end - 6), pre_end + 1):
                    avg_vask += self.history_vask[i]
                    avg_vbid += self.history_vbid[i]
                avg_vask = avg_vask/max(1, min(7, pre_end+1))
                avg_vbid = avg_vbid/max(1, min(7, pre_end+1))



                start = max(0, pre_end - 5)
                grad_bid = ((vbid - self.history_vbid[start])/ (min(len(self.history_vbid), 5)))/ max(self.history_vbid[start], 1)
                grad_ask = ((vask - self.history_vask[start])/ (min(len(self.history_vask), 5)))/ max(self.history_vask[start], 1)

                bslope, bintercept, br, bp, bstd_err = 0, 0, 0, 0, 0
                aslope, aintercept, ar, ap, astd_err = 0, 0, 0, 0, 0
                if len(self.history_vbid) >= 2:
                    bslope, bintercept, br, bp, bstd_err = stats.linregress(list(range(len(self.history_vbid[-15:]))), self.history_vbid[-15:])
                    aslope, aintercept, ar, ap, astd_err = stats.linregress(list(range(len(self.history_vask[-15:]))), self.history_vask[-15:])
                
                if abs(br) >= 0.8:
                    check_bid = bintercept + bslope * len(self.history_vbid) 
                else:
                    if rsi == -1:
                        check_bid = avg_vbid * (1 + grad_bid)
                    else:
                        check_bid = avg_vbid * (1 + grad_bid * (1 + (75 - rsi)/100 * 0.8))

                if abs(ar) >= 0.8:
                    check_ask = aintercept + aslope * len(self.history_vask) 
                else:
                    if rsi == -1:
                        check_ask = avg_vask * (1 + grad_ask)
                    else:
                        check_ask = avg_vask * (1 + grad_ask * (1 + (rsi - 25)/100 * 0.8))

                new_bid_price = min(max(int((check_bid)//TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS), 0), bid_prices[0] + TICK_SIZE_IN_CENTS) if bid_prices[0] != 0 else 0
                new_ask_price = max(max(int((check_ask)//TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS), 0), ask_prices[0] - TICK_SIZE_IN_CENTS) if ask_prices[0] != 0 else 0


                deviation_bid = False
                deviation_ask = False

                if len(self.history_vask) >= 2:                 
                    mslope, mintercept, mr, mp, mstd_err = stats.linregress(list(range(len(self.history_mid[-15:]))), self.history_mid[-15:])
                    
                    if abs(mr) > 0.55:
                        predict_mid = mintercept + mslope * len(self.history_mid)
                        deviation_bid = (predict_mid + abs(mstd_err)) <= new_bid_price
                        deviation_ask = new_ask_price <= (predict_mid - abs(mstd_err))
                    else:
                        predict_mid = -1
                        if len(self.history_vask) >= 2:
                            astd_err = statistics.stdev(self.history_vask[-7:])
                            deviation_ask = new_ask_price <= (avg_vask - abs(astd_err))
                        if len(self.history_vbid) >= 2:
                            bstd_err = statistics.stdev(self.history_vbid[-7:])
                            deviation_bid = (avg_vbid + abs(bstd_err)) <= new_bid_price 

                if self.bid_id != 0 and new_bid_price != 0 and (new_bid_price > self.bid_price * (1+grad_bid) or new_bid_price < self.bid_price * (1-grad_bid)):
                    self.send_cancel_order(self.bid_id)
                    self.bid_id = 0
                    self.bid_ordered = 10
                if self.ask_id != 0 and new_ask_price != 0 and (new_bid_price > self.ask_price * (1+grad_ask) or new_bid_price < self.ask_price * (1-grad_ask)):
                    self.send_cancel_order(self.ask_id)
                    self.ask_id = 0
                    self.ask_ordered = 10

                
                if self.bid_id == 0 and new_bid_price != 0 and (self.bid_ordered + self.position + LOT_SIZE) < POSITION_LIMIT and (not deviation_bid or not over_bought_rsi):
                    self.bid_id = next(self.order_ids)
                    self.bid_price = new_bid_price
                    self.bid_ordered += LOT_SIZE
                    self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                    self.bids.add(self.bid_id)
                
                
                if self.ask_id == 0 and new_ask_price != 0 and (self.position - self.ask_ordered - LOT_SIZE) > (-1 * POSITION_LIMIT) and (not deviation_ask or not over_sold_rsi):
                    self.ask_id = next(self.order_ids)
                    self.ask_price = new_ask_price
                    self.ask_ordered += LOT_SIZE
                    self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                    self.asks.add(self.ask_id)

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)
        if client_order_id in self.bids:
            self.position += volume
            self.bid_ordered -= volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.ask_ordered -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info("received trade ticks for instrument %d with sequence number %d", instrument,
                         sequence_number)
