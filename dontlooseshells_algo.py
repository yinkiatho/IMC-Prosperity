import json
from datamodel import Order, ProsperityEncoder, Symbol, TradingState, Trade, OrderDepth
from typing import Any, Dict, List
import numpy 
import math
import statistics
import jsonpickle
import pandas as pd

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
    'VWAP_BID': [],
    'VWAP_ASK': [],    
    'MID_PRICE': [],
    'MID_PRICE_DIFF': [],
}


class Trader:
    
    df = {'AMETHYSTS': inner_dict, 'STARFRUIT': inner_dict}
    empty_state = {'AMETHYSTS': inner_dict, 'STARFRUIT': inner_dict}
    
    
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
    
    def run(self, state: TradingState):
        #print(self.df)
        print("Timestamp: " + str(state.timestamp))
        print("Observations: " + str(state.observations))
        
        # Decode into df
        try:
        # Decode into df
            if state.traderData == "start":
                self.df = self.empty_state
            else:
                self.df = jsonpickle.decode(state.traderData)
            print(len(self.df['AMETHYSTS']['TIMESTAMP']))
        except json.JSONDecodeError as e:
            print("Initial State")
            self.df = self.empty_state
        

	    # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if product == 'AMETHYSTS':
                amethysts_orders = self.trade_stationary(state, 10000, product)
                result[product] = amethysts_orders
                continue
            else:
                acceptable_price = 1000
                
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(self.df)
        
		# Sample conversion request. Check more details below. 
        conversions = 1
        if len(self.df['AMETHYSTS']['MID_PRICE_DIFF']) > 7:
            print("Testing RSI 7", self.rsi_7(self.df['AMETHYSTS']['MID_PRICE_DIFF']))
        return result, conversions, traderData
    
    
    # Simple function trade around stationary price
    def trade_stationary(self, state: TradingState, acceptable_price: int, product: str):
        orders: list[Order] = []
        
        # Orders to be placed on exchange matching engine
        orders_sell = state.order_depths[product].sell_orders
        orders_buy = state.order_depths[product].buy_orders
        
        if len(orders_sell) != 0:
            min_ask, max_ask, ask_volume = min(orders_sell.keys()), max(orders_sell.keys()), sum(orders_sell.values())
            vwap_ask = 0
            for price, amount in orders_sell.items():
                vwap_ask += int(price) * amount
                if int(price) < acceptable_price:
                    print("BUY", str(-amount) + "x", price)
                    orders.append(Order(product, price, -amount))
                    
            vwap_ask = vwap_ask / ask_volume
                    
        if len(orders_buy) != 0:
            min_bid, max_bid, bid_volume = min(orders_buy.keys()), max(orders_buy.keys()), sum(orders_buy.values())
            vwap_bid = 0
            for price, amount in orders_buy.items():
                vwap_bid += int(price) * amount
                if int(price) > acceptable_price:
                    print("SELL", str(amount) + "x", price)
                    orders.append(Order(product, price, amount))
            vwap_bid = vwap_bid / bid_volume
        
        
        # Append to database
        self.df['AMETHYSTS']['TIMESTAMP'].append(state.timestamp)
        self.df['AMETHYSTS']['MAX_BID'].append(max_bid)
        self.df['AMETHYSTS']['MIN_BID'].append(min_bid)
        self.df['AMETHYSTS']['MAX_ASK'].append(max_ask)
        self.df['AMETHYSTS']['MIN_ASK'].append(min_ask)
        self.df['AMETHYSTS']['BID_VOLUME'].append(bid_volume)
        self.df['AMETHYSTS']['ASK_VOLUME'].append(ask_volume)
        self.df['AMETHYSTS']['VWAP_BID'].append(vwap_bid)
        self.df['AMETHYSTS']['VWAP_ASK'].append(vwap_ask)
        self.df['AMETHYSTS']['MID_PRICE'].append((vwap_bid + vwap_ask) / 2)
        if len(self.df['AMETHYSTS']['MID_PRICE']) > 2:
            self.df['AMETHYSTS']['MID_PRICE_DIFF'].append(((self.df['AMETHYSTS']['MID_PRICE'][-1] - self.df['AMETHYSTS']['MID_PRICE'][-2]) / self.df['AMETHYSTS']['MID_PRICE'][-2]))
        else:
            self.df['AMETHYSTS']['MID_PRICE_DIFF'].append(0)

        return orders
    
    
