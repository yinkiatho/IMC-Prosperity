import json
from datamodel import Order, ProsperityEncoder, Symbol, TradingState, Trade, OrderDepth
from typing import Any, Dict, List
import numpy as np
import math
import statistics
import jsonpickle
import pandas as pd
import collections

class Logger:
    # Set this to true, if u want to create
    # local logs
    local: bool 
    # this is used as a buffer for logs
    # instead of stdout
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local      

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        output = json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True)
        if self.local:
            self.local_logs[state.timestamp] = output
        print(output)

        self.logs = ""

    def compress_state(self, state: TradingState) -> dict[str, Any]:
        listings = []
        for listing in state.listings.values():
            listings.append([listing["symbol"], listing["product"], listing["denomination"]])

        order_depths = {}
        for symbol, order_depth in state.order_depths.items():
            order_depths[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return {
            "t": state.timestamp,
            "l": listings,
            "od": order_depths,
            "ot": self.compress_trades(state.own_trades),
            "mt": self.compress_trades(state.market_trades),
            "p": state.pos,
            "o": state.observations,
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.buyer,
                    trade.seller,
                    trade.price,
                    trade.quantity,
                    trade.timestamp,
                ])

        return compressed

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

# This is provisionary, if no other algorithm works.
# Better to loose nothing, then dreaming of a gain.

inner_dict = {
    'TIMESTAMP': [],
    #'MAX_BID': [],
    #'MIN_BID': [],
    #'MAX_ASK': [],
    #'MIN_ASK': [],
    'BID_VOLUME': [],
    'ASK_VOLUME': [],
    'MID_PRICE': [],
    #'BEST_ASK': [],
    #'BEST_BID': [],
    'MACD': [],
    'EMA_A': [],
    'EMA_B': [],
    'SIGNAL': []
}


class Trader:
    
    df = {'AMETHYSTS': inner_dict, 'STARFRUIT': inner_dict}
    empty_state = {'AMETHYSTS': inner_dict, 'STARFRUIT': inner_dict}
    pos_limits = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    cpos = {'AMETHYSTS': 0, 'STARFRUIT': 0}
    
    # Hyper Parameters
    macd_window = [12, 26 , 9]
    linear_regression = [6, [0.0, 0.9487079673973507, 0.04882953537331608, 0.0, 0.001374535182263223, 0.0], 5.475369188194236]
    SPREAD = 3
    macd_active = False
    MACD_MAX_ORDER = 100000000
    extra=20
    REGRESSION_SPREAD = 1
    regression_extra = 20
    days=3
    price_diff=3
    
    starfruit_cache = []
    starfruit_prices = []
    starfruit_past_bids = []
    starfruit_past_asks = []
    starfruit_ema_a = []
    starfruit_ema_b = []
    starfruit_vwap = []
    
    
    def __init__(self, macd_window=[12, 26 , 9], 
                        linear_regression=[6, [0.0, 0.9487079673973507, 0.04882953537331608, 0.0, 0.001374535182263223, 0.0], 5.475369188194236],
                        SPREAD=3, REGRESSION_SPREAD=1, use_macd=False, MACD_MAX_ORDER = 100000000, extra=20, regression_extra=20,
                        price_diff=3, days=3):
        
        # Hyper Parameters for tuning
        self.linear_regression = linear_regression
        self.macd_window = macd_window
        self.SPREAD = SPREAD
        self.macd_active = use_macd
        self.MACD_MAX_ORDER = MACD_MAX_ORDER
        self.extra = extra
        self.REGRESSION_SPREAD = REGRESSION_SPREAD
        self.regression_extra = regression_extra        
        self.price_diff = price_diff
        self.days = days
    
    
    def calculate_prices(self, days: int):
        # Calculate the average bid and ask price for the last days

        relevant_bids = []
        for bids in self.starfruit_past_bids[-days:]:
            relevant_bids.extend([(value, bids[value]) for value in bids])
        relevant_asks = []
        for asks in self.starfruit_past_asks[-days:]:
            relevant_asks.extend([(value, asks[value]) for value in asks])

        avg_bid = np.average([x[0] for x in relevant_bids], weights=[x[1] for x in relevant_bids])
        avg_ask = np.average([x[0] for x in relevant_asks], weights=[x[1] for x in relevant_asks])

        return avg_bid, avg_ask
            
            
    def calculate_ema2(self, product, windows=[12, 26, 9]):

        for window in windows[:2]:
            K = 2 / (window + 1)
            
            if len(self.starfruit_prices) < window:
                return
            elif len(self.starfruit_prices) == window:
                if window == windows[0]:
                    self.starfruit_ema_a.append(statistics.mean(self.starfruit_prices[:window]))
                elif window == windows[1]:
                    self.starfruit_ema_b.append(statistics.mean(self.starfruit_prices[:window]))
                    
            # len > window
            else:
                if window == windows[0]:
                    self.starfruit_ema_a.append(self.starfruit_prices[-1] * K + ((1-K) *  self.starfruit_ema_a[-1]))
                elif window == windows[1]:
                    self.starfruit_ema_b.append(self.starfruit_prices[-1] * K + ((1-K) * self.starfruit_ema_b[-1]))
                    
        if len( self.starfruit_ema_a) > 0 and len(self.starfruit_ema_b) > 0:
            self.starfruit_macd.append( self.starfruit_ema_a[-1] -  self.starfruit_ema_b[-1])
            
        window = windows[2]
        
        K = 2 / (window + 1)
        if len(self.starfruit_macd) < window:
            return
        elif len(self.starfruit_macd) == window:
            self.starfruit_signal.append(statistics.mean(self.starfruit_macd[:window]))
        else:
            self.starfruit_signal.append(self.starfruit_macd[-1] * K + ((1-K) * self.starfruit_signal[-1]))
                
        #return ema
    
    def z_score(self, df):
        return (df[-1] - statistics.mean(df)) / statistics.stdev(df)
    

    
    def extract_values(self, orders_sell, orders_buy):
        
        min_bid, max_bid, bid_volume, best_bid_volume, best_bid_price, depth = 0, 0, 0, 0, 0, set()
        # match bids with our ask
        for price, amount in orders_buy.items():
            depth.add(price)
            
            if min_bid == 0:
                min_bid = price
            else:
                min_bid = min(price, min_bid)
            
            max_bid = max(price, max_bid)
            bid_volume += amount
            
            if price > best_bid_price or price == best_bid_price and amount > best_bid_volume:
                best_bid_volume = amount
                best_bid_price = price
        
        min_ask, max_ask, ask_volume, best_ask_volume, best_ask_price, ask_depth = 0, 0, 0, 0, math.inf, set()
        for price, amount in orders_sell.items():
            
            ask_depth.add(price)
            if min_ask == 0:
                min_ask = price
            else:
                min_ask = min(price, min_ask)
                
            max_ask = max(price, max_ask)
            ask_volume += amount
                
            if price < best_ask_price or price == best_ask_price and amount > best_ask_volume:
                best_ask_volume = amount
                best_ask_price = price
        
        
        return min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, len(depth), len(ask_depth)
        
    def calc_next_price(self, current_mid):
        #print(self.starfruit_cache)
        coef = self.linear_regression[1]
        intercept = self.linear_regression[2]
        next_price = intercept
        for i in range(0, len(coef)):
            #print("Coefficient: " + str(coef[i]) + " Current Mid Price: " + str(self.starfruit_cache[-(i + 1)]))
            #next_price += coef[i] * self.starfruit_cache[-(i + 1)]
            next_price += coef[i] * self.starfruit_vwap[-(i + 1)]
        
        return next_price
    
    def add_to_df(self, product, data):
        timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price, best_ask_price, best_bid_price = data
        self.df[product]['TIMESTAMP'].append(timestamp)
        #self.df[product]['MAX_BID'].append(max_bid)
        #self.df[product]['MIN_BID'].append(min_bid)
        #self.df[product]['MAX_ASK'].append(max_ask)
        #self.df[product]['MIN_ASK'].append(min_ask)
        self.df[product]['BID_VOLUME'].append(bid_volume)
        self.df[product]['ASK_VOLUME'].append(ask_volume)
        self.df[product]['MID_PRICE'].append(mid_price)
        #self.df[product]['BEST_ASK'].append(best_ask_price)
        #self.df[product]['BEST_BID'].append(best_bid_price)
            
        #if len(self.df[product]['MID_PRICE']) >= 26:
        if product == 'STARFRUIT':
            self.starfruit_prices.append(mid_price)
        #self.calculate_ema2(product, self.macd_window)
            
    def run(self, state: TradingState):

        #print(self.df)
        print("Timestamp: " + str(state.timestamp))
        print("Observations: " + str(state.observations))
        #print("Market Trades: " + str(state.market_trades))
        print("Positions: " + str(state.position))
        
        # Decode into df
        try:
            if state.traderData == "start":
                self.df = self.empty_state
            else:
                self.df = jsonpickle.decode(state.traderData)
            #print(len(self.df['AMETHYSTS']['TIMESTAMP']))
        except json.JSONDecodeError as e:
            print("Initial State")
            self.df = self.empty_state
            
        # Iterating to get current position
        for key, val in state.position.items():
            self.cpos[key] = val
            
        #for key, val in state.pos.items():
            #print("Current Position of " + key + ": " + str(val))
        
        result = {}
        for product in state.order_depths.keys():
            
            if product == 'AMETHYSTS':
                amethysts_orders = self.trade_stationary(state, 10000, product)
                result[product] = amethysts_orders
                #result[product] = []
                
            elif product == 'STARFRUIT':
                if not self.macd_active: 
                    num_lags, coefficients, intercept = self.linear_regression
                    starfruit_orders = self.trade_regression(state, product, num_lags, coefficients, intercept)
                    # Current Best regression
                    #starfruit_orders = self.avellaneda(state, product)                                       
                else:
                    starfruit_orders = self.trade_momentum(state, product)
                    
                result[product] = starfruit_orders
		# String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(self.df)
        
		# Sample conversion request. Check more details below. 
        conversions = 1
        
        return result, conversions, traderData
    
    
    # Simple function trade around stationary price
    def trade_stationary(self, state: TradingState, acceptable_price: int, product: str):
        orders: list[Order] = []
        
        # Orders to be placed on exchange matching engine
        orders_sell = collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items()))
        orders_buy = collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, buy_depth, ask_depth = self.extract_values(orders_sell, orders_buy)
        
        #mid_price = (best_bid_price + best_ask_price) / 2
        mid_price = statistics.median([max_bid, min_ask])
        undercut_buy = best_bid_price + self.SPREAD
        undercut_sell = best_ask_price - self.SPREAD
        
        #if len(self.df[product]['MID_PRICE']) >= 3:
        #    acceptable_price = sum(self.df[product]['MID_PRICE'][-3:]) / 3
        bid_pr = min(undercut_buy, acceptable_price - self.SPREAD) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acceptable_price + self.SPREAD)
        
        
        cpos = self.cpos[product] # eg. cpos of 5 with pos limit of 20 means we can buy 15 more, and sell 25 more of product
        
        if len(orders_buy) != 0:
            for price, amount in orders_buy.items():
                if (price > acceptable_price) and cpos > -self.pos_limits[product]:
                    sell_amount = max(-amount, -self.pos_limits[product] - cpos)
                    #sell_amount = -amount
                    if sell_amount < 0:
                        print("SELL", str(sell_amount) + "x", price)
                        cpos += sell_amount
                        orders.append(Order(product, price, sell_amount))
                        
                        
        if (cpos > -self.pos_limits['AMETHYSTS']) and (self.cpos[product] > 0):
            num = max(-self.extra, -self.pos_limits['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell - self.SPREAD, acceptable_price + self.SPREAD), num))
            cpos += num

        if (cpos > -self.pos_limits['AMETHYSTS']) and (self.cpos[product] < -15):
            num = max(-self.extra, -self.pos_limits['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell + self.SPREAD, acceptable_price + self.SPREAD), num))
            cpos += num

        if cpos > -self.pos_limits['AMETHYSTS']:
            num = max(-self.extra, -self.pos_limits['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num
                                 
                        
        cpos = self.cpos[product]
        
        if len(orders_sell) != 0:
            for price, amount in orders_sell.items():
            # Current lower than predicted next price, so we buy, amount here is negative
                if (price < acceptable_price) and cpos < self.pos_limits[product]:
                    buy_amount = min(-amount, self.pos_limits[product] - cpos)
                    if buy_amount > 0:
                        print("BUY", str(buy_amount) + "x", price)
                        orders.append(Order(product, price, buy_amount))
                        cpos += buy_amount
                        
        
        if (cpos < self.pos_limits['AMETHYSTS']) and (self.cpos[product] < 0):
            num = min(self.extra, self.pos_limits['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + self.SPREAD, acceptable_price - self.SPREAD), num))
            cpos += num

        if (cpos < self.pos_limits['AMETHYSTS']) and (self.cpos[product] > 15):
            num = min(self.extra, self.pos_limits['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - self.SPREAD, acceptable_price - self.SPREAD), num))
            cpos += num

        if cpos < self.pos_limits['AMETHYSTS']:
            num = min(self.extra, self.pos_limits['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        DATA = [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price, best_ask_price, best_bid_price]
        # Append to databasez
        self.add_to_df(product, DATA)
        return orders
    
    
    def trade_pair_arbitrage(self, state: TradingState, product1: str, product2: str):
        order_depths = state.order_depths
        orders = {product1: [], product2: []}
        CONSTANT , coef, STD = 1555, 0.342475, 25 #STD is the standard deviation of the spread
        
        # Orders to be placed on exchange matching engine
        orders1_sell, orders1_buy = collections.OrderedDict(sorted(order_depths[product1].sell_orders.items())), collections.OrderedDict(sorted(order_depths[product1].buy_orders.items(), reverse=True))
        orders2_sell, orders2_buy = collections.OrderedDict(sorted(order_depths[product2].sell_orders.items())), collections.OrderedDict(sorted(order_depths[product2].buy_orders.items(), reverse=True))
        
        min_bid1, max_bid1, bid_volume1, best_bid_volume1, min_sell1, max_sell1, sell_volume1, best_ask_volume1, best_bid_price1, best_ask_price1, buy_depth1, ask_depth1 = self.extract_values(orders1_sell, orders1_buy)
        min_bid2, max_bid2, bid_volume2, best_bid_volume2, min_sell2, max_sell2, sell_volume2, best_ask_volume2, best_bid_price2, best_ask_price2, buy_depth2, ask_depth2 = self.extract_values(orders2_sell, orders2_buy)
        
        mid_price1, mid_price2 = (max_bid1 + max_sell1) / 2, (max_bid2 + max_sell2) / 2
        
        ideal_price2 = mid_price1 * coef + CONSTANT
        diff = mid_price2 - ideal_price2
        
        cpos1, cpos2 = self.cpos[product1], self.cpos[product2]
    
        
        if diff >= STD: # Arbitrage opportunity, current middle price for product 2 is higher than expected, so we sell product 2 and buy product 1
            print("Arbitrage opportunity: Buying " + product1 + " and selling " + product2)
            for price, amount in orders2_buy.items():
                if ((price >= ideal_price2 + STD) or ((cpos2 > 0) and (price == ideal_price2 + STD))) and cpos2 > -self.pos_limits[product2]:  
                    sell_amount = max(-amount, -self.pos_limits[product2] - cpos2)
                    if sell_amount < 0:
                        print("SELL", str(-amount) + "x", price)
                        cpos2 -= amount
                        orders.append(Order(product2, price, sell_amount))
            
            for price, amount in orders1_sell.items():
                if ((price <= mid_price1 - STD) or ((cpos1 < 0) and (price == mid_price1 - STD))) and cpos1 < self.pos_limits[product1]:
                    buy_amount = min(amount, self.pos_limits[product1] - cpos1)
                    if buy_amount > 0:
                        print("BUY", str(amount) + "x", price)
                        orders.append(Order(product1, price, buy_amount))
                        cpos1 += amount
    
            
        elif diff <= -STD: # Arbitrage opportunity, current middle price for product 2 is lower than expected, so we sell product 1 and buy product 2
            print("Arbitrage opportunity: Buying " + product2 + " and selling " + product1)
            for price, amount in orders1_buy.items():
                if ((price >= mid_price1 + STD) or ((cpos1 > 0) and (price ==  mid_price1 + STD))) and cpos1 > -self.pos_limits[product1]:  
                    sell_amount = max(-amount, -self.pos_limits[product1] - cpos1)
                    if sell_amount < 0:
                        print("SELL", str(-amount) + "x", price)
                        cpos1 -= amount
                        orders.append(Order(product1, price, sell_amount))
                        
                        
            for price, amount in orders2_sell.items():
                if ((price <= ideal_price2 - STD) or ((cpos2 < 0) and (price == ideal_price2 - STD))) and cpos2 < self.pos_limits[product2]:
                    buy_amount = min(amount, self.pos_limits[product2] - cpos2)
                    if buy_amount > 0:
                        print("BUY", str(amount) + "x", price)
                        orders.append(Order(product2, price, buy_amount))
                        cpos2 += amount
                    
                
        self.add_to_df(product1, [state.timestamp, max_bid1, min_bid1, max_sell1, min_sell1, bid_volume1, sell_volume1, mid_price1, best_ask_price1, best_bid_price1])
        self.add_to_df(product2, [state.timestamp, max_bid2, min_bid2, max_sell2, min_sell2, bid_volume2, sell_volume2, mid_price2, best_ask_price2, best_bid_price2])
        
        return orders
           
    
    def trade_regression(self, state: TradingState, product: str, N: int, coefficients: List[float], intercept: float) -> None:
        
        orders: list[Order] = []
        
        # Orders to be placed on exchange matching engine
        orders_sell = collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items()))
        orders_buy = collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, buy_depth, ask_depth = self.extract_values(orders_sell, orders_buy)
        
        mid_price = statistics.median([max_bid, min_ask])
        vwap_buy, vwap_sell = [price * amount for price, amount in orders_buy.items()], [price * -amount for price, amount in orders_sell.items()]
        vwap = sum(vwap_buy + vwap_sell) / (sum([amount for price, amount in orders_buy.items()])  - sum([amount for price, amount in orders_sell.items()]))
        
        
        #Skip if not enough data
        if len(self.starfruit_vwap) < N:
            self.starfruit_vwap.append(vwap)
            #self.starfruit_cache.append(mid_price)
        else:
            #self.starfruit_cache.append(mid_price)
            #self.starfruit_cache.pop(0)
            self.starfruit_vwap.append(vwap)
            self.starfruit_vwap.pop(0)
            
        self.starfruit_past_asks.append(orders_sell)
        self.starfruit_past_bids.append(orders_buy)
        
        self.add_to_df(product, [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price, best_ask_price, best_bid_price])
        if len(self.starfruit_vwap) < N:
            return orders
        
        
        # Calculate regression
        #ideal_mid_price = self.calc_next_price(mid_price)
        ideal_mid_price = self.calc_next_price(vwap)
        ideal_mid_price = int(round(ideal_mid_price, 2))
    
        undercut_buy = best_bid_price + 1 # best_bid price is the best price where others want to buy from us, 
        undercut_sell = best_ask_price - 1 # best_ask price is the best price where others want to sell to us
        
        days = 3
        min_price_diff = 3

        bid_pr = min(undercut_buy, ideal_mid_price - self.REGRESSION_SPREAD) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, ideal_mid_price + self.REGRESSION_SPREAD)
        
        avg_bid, avg_ask = self.calculate_prices(days)
        num_sold, num_bought = 0, 0
        cpos = self.cpos[product]

        for price, amount in orders_buy.items():
            # Current price is greater than predicted next price, so we sell, amount here is positive
            if (price >= ideal_mid_price + self.REGRESSION_SPREAD and cpos > -self.pos_limits[product]):
                #or (cpos > 0 and price+1 == ideal_mid_price + self.REGRESSION_SPREAD)):
              #if (price <= ideal_mid_price - self.SPREAD and cpos > -self.pos_limits[product]):
                sell_amount = max(-amount, -self.pos_limits[product] - cpos)
            
                if sell_amount < 0:
                    print("SELL", str(sell_amount) + "x", price)
                    cpos += sell_amount
                    num_sold += 1
                    orders.append(Order(product, price, sell_amount))
        
        
        if (cpos > -self.pos_limits[product]):
            i = num_sold
            while i > len(orders_buy) and cpos > -self.pos_limits[product] and avg_ask - orders_buy[i][0] <= min_price_diff:
                num = min(-self.pos_limits[product]-cpos, -orders_buy[i][1])
                orders.append(Order(product, orders_buy[i][0], num))
                cpos += num
                i += 1

        if (cpos > -self.pos_limits[product]):
            num = -self.pos_limits[product]-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        cpos = self.cpos[product]
        
        for price, amount in orders_sell.items():
            # Current lower than predicted next price, so we buy, amount here is negative
            if (price <= ideal_mid_price - self.REGRESSION_SPREAD and cpos < self.pos_limits[product]):
                #or (cpos < 0 and price-1 == ideal_mid_price - self.REGRESSION_SPREAD)):
            #if (price >= ideal_mid_price + self.SPREAD and cpos < self.pos_limits[product]):
                buy_amount = min(-amount, self.pos_limits[product] - cpos)
                if buy_amount > 0:
                    print("BUY", str(buy_amount) + "x", price)
                    orders.append(Order(product, price, buy_amount))
                    cpos += buy_amount
                    num_bought += 1
        
         
        if (cpos < self.pos_limits[product]):
            i = num_bought
            while i > len(orders_sell) and cpos < self.pos_limits[product] and orders_sell[i][0] - avg_bid  <= min_price_diff:
                num = max(self.pos_limits[product]-cpos, -orders_sell[i][1])
                orders.append(Order(product, orders_sell[i][0], num))
                cpos += num
                i += 1
                
        if cpos < self.pos_limits[product]:
            num = self.pos_limits[product] - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
            
        
            
        #self.add_to_df(product, [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price, best_ask_price, best_bid_price])
        return orders
    
    
    def trade_momentum(self, state: TradingState, product:str):
        
        orders = []
        # Orders to be placed on exchange matching engine
        orders_sell = collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items()))
        orders_buy = collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        
        min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, buy_depth, ask_depth = self.extract_values(orders_sell, orders_buy)
        #mid_price = (best_ask_price + best_bid_price) / 2
        mid_price = statistics.median([max_bid, min_ask])

        
        cpos = self.cpos[product]
        
        # Generate Signals calculate difference between 26EMA and 12EMA, signal line 9EMA
        self.add_to_df(product, [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price, best_ask_price, best_bid_price])
        
        if len(self.starfruit_signal) == 0:
            return orders
        
        MACD = self.starfruit_macd[-1]
        SIGNAL = self.starfruit_signal[-1]

        print(f"MACD: {MACD} SIGNAL: {SIGNAL} at {state.timestamp}")
        #print(state.own_trades)
        
        # Buy Signal, MACD > SIGNAL
        if MACD > SIGNAL:
            curr_bought = 0
            for price, amount in orders_sell.items():
                if cpos < self.pos_limits[product]:
                    #buy_amount = min(-amount, self.pos_limits[product] - cpos, ORDER_LIMIT - curr_bought)
                    buy_amount = min(-amount, self.pos_limits[product] - cpos, self.MACD_MAX_ORDER)
                    if buy_amount > 0:
                        print("BUY", str(buy_amount) + "x", price)
                        orders.append(Order(product, price, buy_amount))
                        cpos += buy_amount
                        curr_bought += buy_amount
                        
            #Predicted it will decrease and hence we will LONG
            if cpos < self.pos_limits[product]:
                buy_amount = self.pos_limits[product] - cpos
                buy_pr = max(best_ask_price - 1, mid_price - 1)
                print("BUY", str(buy_amount) + "x", buy_pr)
                orders.append(Order(product, buy_pr, buy_amount))
                cpos += buy_amount
                        
        # Sell Signal, MACD < SIGNAL
        elif MACD < SIGNAL:
            curr_sold = 0
            for price, amount in orders_buy.items():
                if cpos > -self.pos_limits[product]:
                    #sell_amount = max(-amount, -self.pos_limits[product] - cpos, -ORDER_LIMIT - curr_sold)
                    sell_amount = max(-amount, -self.pos_limits[product] - cpos, -self.MACD_MAX_ORDER)
                    if sell_amount < 0:
                        print("SELL", str(sell_amount) + "x", price)
                        orders.append(Order(product, price, sell_amount))
                        cpos += sell_amount
                        curr_sold += sell_amount # sell_amount is negative
                        
                        
            #Predicted it will decrease and hence we will 
            if cpos > -self.pos_limits[product]:
                sell_pr = min(best_bid_price + 1, mid_price + 1)
                sell_amount = -self.pos_limits[product] - cpos
                print("SELL", str(sell_amount) + "x", sell_pr)
                orders.append(Order(product, sell_pr, sell_amount))
                cpos += sell_amount
                        
        return orders
    
    
    
    def calculate_reservation_prices(self, mid_price, PARAMS, product:str):
        #PARAMS['w'] = 0.5 * (PARAMS['y'] ** 2) * (PARAMS['std'] **2) * (self.pos_limits[product] + 1)**2
        PARAMS['w'] = 20
        ask_price = mid_price + (1/PARAMS['y']) * math.log(1 + ((1 - 2 * PARAMS['q']) * (PARAMS['y']**2) * PARAMS['std']**2)/(2*PARAMS['w'] - (PARAMS['y'] ** 2) * (PARAMS['q'] ** 2) * PARAMS['std']**2))
        bid_price = mid_price + (1/PARAMS['y']) * math.log(1 + ((-1 - 2*PARAMS['q']) * (PARAMS['y']**2) * PARAMS['std']**2)/(2*PARAMS['w'] - (PARAMS['y'] ** 2) * (PARAMS['q'] ** 2) * PARAMS['std']**2))
        
        print(f"Reservation Bid Price: {bid_price} Reservation Ask Price: {ask_price} at {mid_price} for {product}, Average Reservation Price: {(ask_price + bid_price)/2}")
        return int(bid_price), int(ask_price)
    
    
    
    def avellaneda(self, state: TradingState, product:str):
        
        orders = []
        # Orders to be placed on exchange matching engine
        orders_sell = collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items()))
        orders_buy = collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        
        min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, buy_depth, ask_depth = self.extract_values(orders_sell, orders_buy)
        mid_price = (max_bid + min_ask) / 2
        
        print(f"Current Mid Price: {mid_price} at {state.timestamp}, Min Bid: {min_bid}, Max Bid: {max_bid}, Min Ask: {min_ask}, Max Ask: {max_ask}, Bid Volume: {bid_volume}, Ask Volume: {ask_volume}, Best Ask Price: {best_ask_price}, Best Bid Price: {best_bid_price}")
        cpos = self.cpos[product]
        
        if len(self.starfruit_prices) < 5:
            self.add_to_df(product, [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price, best_ask_price, best_bid_price])
            return orders

        
        
        # Parameters here, Reservation Price, Spread
        PARAMS = {
            "s": mid_price,
            "q": cpos,
            "y": 1,
            "std": statistics.stdev(self.starfruit_prices[-5:])
        }
        # Reservation Bid Price is the price at which we are willing to buy one more asset, Reservation Ask Price is the price at which we are willing to sell one more asset
        reservation_bid, reservation_ask = self.calculate_reservation_prices(mid_price, PARAMS, product)
        if reservation_bid < 0 or reservation_ask < 0:
            return orders
        
        # Adjust downwards:
        reservation_bid = reservation_bid - 1
        reservation_ask = reservation_ask + 1
        
        # Temporary using Stationary Trading Method
        num_sell = 0
        if len(orders_buy) != 0:
            for price, amount in orders_buy.items():
                if (price > reservation_ask) and cpos > -self.pos_limits[product]:
                    sell_amount = max(-amount, -self.pos_limits[product] - cpos)
                    #sell_amount = -amount
                    if sell_amount < 0:
                        print("SELL", str(sell_amount) + "x", price)
                        cpos += sell_amount
                        # cpos -= amount
                        orders.append(Order(product, price, sell_amount))
                        num_sell += sell_amount
                        
        cpos = self.cpos[product]
        
        num_buy = 0
        if len(orders_sell) != 0:
            for price, amount in orders_sell.items():
            # Current lower than predicted next price, so we buy, amount here is negative
                if (price < reservation_bid) and cpos < self.pos_limits[product]:
                    buy_amount = min(-amount, self.pos_limits[product] - cpos)
                    if buy_amount > 0:
                        print("BUY", str(buy_amount) + "x", price)
                        orders.append(Order(product, price, buy_amount))
                        #cpos += amount
                        cpos += buy_amount
                        num_buy += buy_amount
                        
        # Current Extra
        current = num_buy + num_sell + self.cpos[product]
        if current > 0:
            extra = self.pos_limits[product] - current
        else:
            extra = -self.pos_limits[product] - current
            extra = -extra
            
        # Fill up orders
        orders.append(Order(product, max(int(reservation_ask), best_bid_price + 1), -extra))
        orders.append(Order(product, min(int(reservation_bid), best_ask_price - 1), extra))
        
        DATA = [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, (max_bid + min_ask) / 2, best_ask_price, best_bid_price]
        # Append to database
        self.add_to_df(product, DATA)
        return orders        
    
        # Adjust Reservation Price based on current position, 
        # If Positive Inventory, researvation price < midprice, else > midprice
        
        
        
        # Adjust Spread based on Order-Book Liquidity 
        
    
        

            
        
        
                                                        
                                                        
            
        
        
        
            
        
        
            
    
    
        
            
            
    
            
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        

        
    
    
    
    
    
    
