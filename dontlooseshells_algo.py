import json
from datamodel import Order, ProsperityEncoder, Symbol, TradingState, Trade, OrderDepth
from typing import Any, Dict, List
import numpy 
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
            "p": state.position,
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
    'MAX_BID': [],
    'MIN_BID': [],
    'MAX_ASK': [],
    'MIN_ASK': [],
    'BID_VOLUME': [],
    'ASK_VOLUME': [],
    'MID_PRICE': [],
    'MID_PRICE_DIFF': [],
}


class Trader:
    
    df = {'AMETHYSTS': inner_dict, 'STARFRUIT': inner_dict}
    empty_state = {'AMETHYSTS': inner_dict, 'STARFRUIT': inner_dict}
    pos_limits = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    cpos = {'AMETHYSTS': 0, 'STARFRUIT': 0}
    
    
    # Takes in 'MID_PRICE_DIFF'
    def rsi_7(self, df):
        last_7 = df[-7:]
        gain = sum([x for x in last_7 if x > 0]) / 7
        loss = -1 * sum([x for x in last_7 if x < 0]) / 7
        
        print("Gain: " + str(gain) + ", Loss: " + str(loss))
        if loss == 0:
            return 100
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def z_score(self, df):
        return (df[-1] - statistics.mean(df)) / statistics.stdev(df)
    
    def add_to_df(self, product, data):
        timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price = data
        self.df[product]['TIMESTAMP'].append(timestamp)
        self.df[product]['MAX_BID'].append(max_bid)
        self.df[product]['MIN_BID'].append(min_bid)
        self.df[product]['MAX_ASK'].append(max_ask)
        self.df[product]['MIN_ASK'].append(min_ask)
        self.df[product]['BID_VOLUME'].append(bid_volume)
        self.df[product]['ASK_VOLUME'].append(ask_volume)
        self.df[product]['MID_PRICE'].append(mid_price)
        if len(self.df[product]['MID_PRICE']) > 2:
            self.df[product]['MID_PRICE_DIFF'].append(((self.df[product]['MID_PRICE'][-1] - self.df[product]['MID_PRICE'][-2]) / self.df[product]['MID_PRICE'][-2]))
        else:
            self.df[product]['MID_PRICE_DIFF'].append(0)
    
    def run(self, state: TradingState):

        #print(self.df)
        print("Timestamp: " + str(state.timestamp))
        print("Observations: " + str(state.observations))
        #print("Market Trades: " + str(state.market_trades))
        #rint("Own Trades: " + str(state.own_trades))
        
        # Decode into df
        try:
            if state.traderData == "start":
                self.df = self.empty_state
            else:
                self.df = jsonpickle.decode(state.traderData)
            print(len(self.df['AMETHYSTS']['TIMESTAMP']))
        except json.JSONDecodeError as e:
            print("Initial State")
            self.df = self.empty_state
            
        # Iterating to get current position
        for key, val in state.position.items():
            self.cpos[key] = val
            
        for key, val in state.position.items():
            print("Current Position of " + key + ": " + str(val))
        
        result = {}
        for product in ['AMETHYSTS', 'STARFRUIT']:
            if product == 'AMETHYSTS':
                amethysts_orders = self.trade_stationary(state, 10000, product)
                result[product] = amethysts_orders

            elif product == 'STARFRUIT':
                starfruit_orders = self.trade_regression(state, product, 10, [0.26246044, 0.16805252, 0.17344203, 0.12245118, 0.08862426,
                                                                              0.03932866, 0.03248221, 0.00336833, 0.04446871, 0.06437383], 2.140)
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
        
        
        cpos = self.cpos[product] # eg. cpos of 5 with pos limit of 20 means we can buy 15 more, and sell 25 more of product
        
        if len(orders_buy) != 0:
            min_bid, max_bid, bid_volume = min(orders_buy.keys()), max(orders_buy.keys()), sum(orders_buy.values())
            for price, amount in orders_buy.items():
                if ((price >= acceptable_price + 1) or ((cpos > 0) and (price == acceptable_price + 1))) and cpos > -self.pos_limits[product]:
                    sell_amount = max(-amount, -self.pos_limits[product] - cpos)
                    if sell_amount < 0:
                        print("SELL", str(-amount) + "x", price)
                        cpos -= amount
                        orders.append(Order(product, price, sell_amount))
                        
        cpos = self.cpos[product]
        
        if len(orders_sell) != 0:
            min_ask, max_ask, ask_volume = min(orders_sell.keys()), max(orders_sell.keys()), sum(orders_sell.values())
            for price, amount in orders_sell.items():
            # Current lower than predicted next price, so we buy
                if ((price <= acceptable_price - 1) or ((cpos < 0) and (price == acceptable_price - 1))) and cpos < self.pos_limits[product]:
                    buy_amount = min(amount, self.pos_limits[product] - cpos)
                    if buy_amount > 0:
                        print("BUY", str(amount) + "x", price)
                        orders.append(Order(product, price, buy_amount))
                        cpos += amount
                        
        
        
        DATA = [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, (max_bid + min_ask) / 2]
        # Append to database
        self.add_to_df(product, DATA)
        return orders
    
    
    def trade_pair_arbitrage(self, state: TradingState, product1: str, product2: str):
        order_depths = state.order_depths
        orders = {product1: [], product2: []}
        CONSTANT , coef, STD = 1555, 0.342475, 25 #STD is the standard deviation of the spread
        
        # Orders to be placed on exchange matching engine
        orders1_sell, orders1_buy = collections.OrderedDict(sorted(order_depths[product1].sell_orders.items())), collections.OrderedDict(sorted(order_depths[product1].buy_orders.items(), reverse=True))
        orders2_sell, orders2_buy = collections.OrderedDict(sorted(order_depths[product2].sell_orders.items())), collections.OrderedDict(sorted(order_depths[product2].buy_orders.items(), reverse=True))
        min_bid1, max_bid1, bid_volume1 = min(orders1_buy.keys()), max(orders1_buy.keys()), sum(orders1_buy.values())
        min_bid2, max_bid2, bid_volume2 = min(orders2_buy.keys()), max(orders2_buy.keys()), sum(orders2_buy.values())
        min_sell1, max_sell1, sell_volume1 = min(orders1_sell.keys()), max(orders1_sell.keys()), sum(orders1_sell.values())
        min_sell2, max_sell2, sell_volume2 = min(orders2_sell.keys()), max(orders2_sell.keys()), sum(orders2_sell.values())
        
        mid_price1, mid_price2 = (max_bid1 + min_sell1) / 2, (max_bid2 + min_sell2) / 2
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
                    
                
        self.add_to_df(product1, [state.timestamp, max_bid1, min_bid1, max_sell1, min_sell1, bid_volume1, sell_volume1, mid_price1])
        self.add_to_df(product2, [state.timestamp, max_bid2, min_bid2, max_sell2, min_sell2, bid_volume2, sell_volume2, mid_price2])
        
        return orders
                        
    
    def trade_regression(self, state: TradingState, product: str, N: int, coefficients: List[float], intercept: float) -> None:
        
        orders: list[Order] = []
        
        # Orders to be placed on exchange matching engine
        orders_sell = collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items()))
        orders_buy = collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        
        min_bid, max_bid, bid_volume = min(orders_buy.keys()), max(orders_buy.keys()), sum(orders_buy.values())
        min_ask, max_ask, ask_volume = min(orders_sell.keys()), max(orders_sell.keys()), sum(orders_sell.values())
        mid_price = (max_bid + min_ask) / 2
        # Skip if not enough data
        if len(self.df[product]['MID_PRICE']) < N:
            return orders
        
        cpos = self.cpos[product]
        
        # Calculate regression
        ideal_mid_price = intercept + sum([self.df[product]['MID_PRICE'][-(i + 1)] * coefficients[i] for i in range(1, N)]) + coefficients[0] * mid_price
        
        for price, amount in orders_buy.items():
            # Current price is greater than predicted next price, so we sell
            if ((price >= ideal_mid_price + 1) or ((cpos > 0) and (price == ideal_mid_price + 1))) and cpos > -self.pos_limits[product]:
                sell_amount = max(-amount, -self.pos_limits[product] - cpos)
                if sell_amount < 0:
                    print("SELL", str(-amount) + "x", price)
                    cpos -= amount
                    orders.append(Order(product, price, sell_amount))
        
        cpos = self.cpos[product]
        
        for price, amount in orders_sell.items():
            # Current lower than predicted next price, so we buy
            if ((price <= ideal_mid_price - 1) or ((cpos < 0) and (price == ideal_mid_price - 1))) and cpos < self.pos_limits[product]:
                buy_amount = min(amount, self.pos_limits[product] - cpos)
                if buy_amount > 0:
                    print("BUY", str(amount) + "x", price)
                    orders.append(Order(product, price, buy_amount))
                    cpos += amount
    
                                
        self.add_to_df(product, [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price])
        return orders
        
            
            
    
            
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        

        
    
    
    
    
    
    
