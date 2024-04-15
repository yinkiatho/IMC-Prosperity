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
    
    #df = {'AMETHYSTS': inner_dict, 'STARFRUIT': inner_dict}
    empty_state = {'AMETHYSTS': inner_dict, 'STARFRUIT': inner_dict, 'ORCHIDS': inner_dict}
    pos_limits = {'AMETHYSTS': 20, 'STARFRUIT': 20, 'ORCHIDS':100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60}
    cpos = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0}
    
    # Hyper Parameters
    linear_regression = [6, [0.0, 0.9487079673973507, 0.04882953537331608, 0.0, 0.001374535182263223, 0.0], 5.475369188194236]
    
    # lag, total_tariff, sunlight, ideal humidity boolean, humidity away from ideal
    orchids_regression =  [1.00005579, -0.0284300701, -0.0000203225,  0.11016487, 0.0115631195]
    SPREAD = 3
    extra=20
    REGRESSION_SPREAD = 1
    regression_extra = 20
    days=3
    price_diff=3
    lower_bound=0
    upper_bound=15
    
    ORCHIDS_LIMIT = 100
    
    #starfruit_prices = []
    starfruit_vwap = []
    orchids_last = 0
    orchids_last_obs = []
    orchids_cache = []
    counter = 0
    counter_beneath_threshold = 0
    start_counter = 0
    humidity_cache = []
    sunlight_cache = [] 
    basket_std = 76
    
    
    chocolate_cache = []
    strawberries_cache = []
    roses_cache = []    
    gift_basket_cache = []
    
    
    def __init__(self, linear_regression=[6, [0.0, 0.9487079673973507, 0.04882953537331608, 0.0, 0.001374535182263223, 0.0], 5.475369188194236],
                        SPREAD=3, REGRESSION_SPREAD=1, extra=20, regression_extra=20, lower_bound=0, upper_bound=15, ORCHIDS_LIMIT=200,
                        price_diff=3, days=3):
        
        # Hyper Parameters for tuning
        self.linear_regression = linear_regression
        self.SPREAD = SPREAD
        self.extra = extra
        self.REGRESSION_SPREAD = REGRESSION_SPREAD
        self.regression_extra = regression_extra        
        self.price_diff = price_diff
        self.days = days
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.orchids_last = 0
        self.orchids_last_obs = []
        self.ORCHIDS_LIMIT = ORCHIDS_LIMIT
        self.buy = -0.2
        self.sell = 0.2
        self.orchids_conversions = 0            
        
    
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
                
            if price < best_ask_price or price == best_ask_price and amount < best_ask_volume:
                best_ask_volume = amount
                best_ask_price = price
        
        
        return min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, len(depth), len(ask_depth)
        
    def calc_next_price(self, current_mid):
        #print(self.starfruit_cache)
        coef = self.linear_regression[1]
        intercept = self.linear_regression[2]
        next_price = intercept
        for i in range(0, len(coef)):

            next_price += coef[i] * self.starfruit_vwap[-(i + 1)]
        
        return next_price
    
    def add_to_df(self, product, data):
        timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price, best_ask_price, best_bid_price = data
        return
            
    def run(self, state: TradingState):

        #print(self.df)
        print("Timestamp: " + str(state.timestamp))
        print("Observations: " + str(state.observations))
        #print("Market Trades: " + str(state.market_trades))
        print("Positions: " + str(state.position))
        
        # Iterating to get current position
        for key, val in state.position.items():
            self.cpos[key] = val
        
        result = self.trade_basket(state)
        for product in state.order_depths.keys():
            
            if product == 'AMETHYSTS':
                amethysts_orders = self.trade_stationary(state, 10000, product)
                result[product] = amethysts_orders
                #result[product] = []
                
            elif product == 'STARFRUIT':
                num_lags, coefficients, intercept = self.linear_regression
                starfruit_orders = self.trade_regression(state, product, num_lags, coefficients, intercept)    
                result[product] = starfruit_orders
                #result[product] = []
                
            elif product == 'ORCHIDS':
                orchids_orders, conversions = self.trade_orchids_arbitrage(state, product)
                result[product] = orchids_orders
            else:
                continue        
                
        return result, self.orchids_conversions, ""
    
    
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
                        
                        
        if (cpos > -self.pos_limits['AMETHYSTS']) and (self.cpos[product] > self.lower_bound):
            num = max(-self.extra, -self.pos_limits['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell - self.SPREAD, acceptable_price + self.SPREAD), num))
            cpos += num

        if (cpos > -self.pos_limits['AMETHYSTS']) and (self.cpos[product] < -self.upper_bound):
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
                        
        
        if (cpos < self.pos_limits['AMETHYSTS']) and (self.cpos[product] < self.lower_bound):
            num = min(self.extra, self.pos_limits['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + self.SPREAD, acceptable_price - self.SPREAD), num))
            cpos += num

        if (cpos < self.pos_limits['AMETHYSTS']) and (self.cpos[product] > self.upper_bound):
            num = min(self.extra, self.pos_limits['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - self.SPREAD, acceptable_price - self.SPREAD), num))
            cpos += num

        if cpos < self.pos_limits['AMETHYSTS']:
            num = min(self.extra, self.pos_limits['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        DATA = [state.timestamp, max_bid, min_bid, max_ask, min_ask, bid_volume, ask_volume, mid_price, best_ask_price, best_bid_price]
        # Append to databasez
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
           
    
    def trade_regression(self, state: TradingState, product: str, N: int, coefficients: List[float], intercept: float, ideal_price=None) -> None:
        
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
        else:
            self.starfruit_vwap.append(vwap)
            self.starfruit_vwap.pop(0)

        if len(self.starfruit_vwap) < N:
            return orders
        
        # Calculate regression
        #target_price = self.calc_next_price(mid_price)
        if ideal_price != None:
            target_price = ideal_price
        else:
            target_price = self.calc_next_price(vwap)
            target_price = int(round(target_price, 2))
    
        undercut_buy = best_bid_price + 1 # best_bid price is the best price where others want to buy from us, 
        undercut_sell = best_ask_price - 1 # best_ask price is the best price where others want to sell to us
        

        bid_pr = min(undercut_buy, target_price - self.REGRESSION_SPREAD) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, target_price + self.REGRESSION_SPREAD)
        
        num_sold, num_bought = 0, 0
        cpos = self.cpos[product]
        
        for price, amount in orders_buy.items():
            # Current price is greater than predicted next price, so we sell, amount here is positive
            if (price >= target_price + self.REGRESSION_SPREAD and cpos > -self.pos_limits[product]):
                #or (cpos > 0 and price+1 == target_price + self.REGRESSION_SPREAD)):
              #if (price <= target_price - self.SPREAD and cpos > -self.pos_limits[product]):
                sell_amount = max(-amount, -self.pos_limits[product] - cpos)
            
                if sell_amount < 0:
                    print("SELL", str(sell_amount) + "x", price)
                    cpos += sell_amount
                    num_sold += 1
                    orders.append(Order(product, price, sell_amount))

        if (cpos > -self.pos_limits[product]):
            num = -self.pos_limits[product]-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num


        cpos = self.cpos[product]
        
        for price, amount in orders_sell.items():
            # Current lower than predicted next price, so we buy, amount here is negative
            if (price <= target_price - self.REGRESSION_SPREAD and cpos < self.pos_limits[product]):
                #or (cpos < 0 and price-1 == target_price - self.REGRESSION_SPREAD)):
            #if (price >= target_price + self.SPREAD and cpos < self.pos_limits[product]):
                buy_amount = min(-amount, self.pos_limits[product] - cpos)
                if buy_amount > 0:
                    print("BUY", str(buy_amount) + "x", price)
                    orders.append(Order(product, price, buy_amount))
                    cpos += buy_amount
                    num_bought += 1
        
                
        if cpos < self.pos_limits[product]:
            num = self.pos_limits[product] - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        return orders
    
    
    def predict_regression_humidity(self):
        X = np.array([i for i in range(len(self.humidity_cache))])
        y = np.array(self.humidity_cache)

        A = np.vstack([X, np.ones(len(X))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        return int(round(len(self.humidity_cache) * m + c))
    
    
    def predict_regression_sunlight(self):
        X = np.array([i for i in range(len(self.sunlight_cache))])
        y = np.array(self.sunlight_cache)

        A = np.vstack([X, np.ones(len(X))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        return int(round(len(self.sunlight_cache) * m + c))
    
    
    def predict_regression_orchids(self):
        X = np.array([i for i in range(len(self.orchids_cache))])
        y = np.array(self.orchids_cache)

        A = np.vstack([X, np.ones(len(X))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        return int(round(len(self.orchids_cache) * m + c))
    
    
    def trade_orchids_arbitrage(self, state: TradingState, product: str):
        
        orders: list[Order] = []
        
        # Orders to be placed on exchange matching engine
        orders_sell = collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items()))
        orders_buy = collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
        min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, buy_depth, ask_depth = self.extract_values(orders_sell, orders_buy)
        
        mid_price = statistics.median([max_bid, min_ask])
        orchid_obs = state.observations.conversionObservations['ORCHIDS']
        
        bid_pr, ask_pr, transportFees, exportTariff, importTariff, sunlight, humidity = orchid_obs.bidPrice, orchid_obs.askPrice, orchid_obs.transportFees, orchid_obs.exportTariff, orchid_obs.importTariff, orchid_obs.sunlight, orchid_obs.humidity
        if len(self.orchids_cache) < 5:
            self.orchids_cache.append(mid_price)
            self.humidity_cache.append(humidity)
            self.sunlight_cache.append(sunlight)
        else:
            self.orchids_cache.append(mid_price)
            self.orchids_cache.pop(0)
            self.humidity_cache.append(humidity)
            self.humidity_cache.pop(0)
            self.sunlight_cache.append(sunlight)
            self.sunlight_cache.pop(0)
        
        
                
        ask_pr = int(ask_pr + transportFees + importTariff)
        bid_pr = int(bid_pr - transportFees - exportTariff)


        predicted_mid_price = self.predict_regression_orchids()
        # Fulfill any orchids where we can sell here and buy from other islands
        cpos = self.cpos[product]
        total_sell_amount = 0
        if best_bid_price > ask_pr:
            sell_amount = max(-best_bid_volume, -self.pos_limits[product] - cpos)
            cpos += sell_amount
            #sell_amount = -best_bid_volume
            if sell_amount < 0:
                orders.append(Order(product, best_bid_price, sell_amount))
                total_sell_amount += sell_amount
            
        for price, amount in orders_buy.items():
            # Current price is greater than predicted next price, so we sell, amount here is positive
            if (price > ask_pr + 1 and cpos > -self.pos_limits[product]):
                #or (cpos > 0 and price+1 == target_price + self.REGRESSION_SPREAD)):
              #if (price <= target_price - self.SPREAD and cpos > -self.pos_limits[product]):
                sell_amount = max(-amount, -self.pos_limits[product] - cpos)
                if sell_amount < 0:
                    print("SELL", str(sell_amount) + "x", price)
                    cpos += sell_amount
                    total_sell_amount += sell_amount
                    orders.append(Order(product, price, sell_amount))
                    
            
        if isinstance(state.position.get('ORCHIDS'), int) and total_sell_amount != 0:
            self.orchids_conversions = -state.position.get('ORCHIDS') - -total_sell_amount
            
        
            
        # Buy Orchids here, sell to other islands
        
        # Fulfill any orchids where we can buy here and sell to other islands
        cpos = self.cpos[product]
        total_buy_amount = 0
        
                    
        if bid_pr > best_ask_price:
            #buy_amount = -best_ask_volume
            buy_amount = min(-best_ask_volume, self.pos_limits[product] - cpos)
            cpos += buy_amount
            # buy_amount = max(-best_ask_volume, self.pos_limits[product] - cpos)
            if buy_amount > 0:
                orders.append(Order(product, best_ask_price, buy_amount))
                total_buy_amount += buy_amount

            
        for price, amount in orders_sell.items():
            # Current price is greater than predicted next price, so we sell, amount here is positive
            if (price < bid_pr - 1 and cpos > -self.pos_limits[product]):
                #or (cpos > 0 and price+1 == target_price + self.REGRESSION_SPREAD)):
              #if (price <= target_price - self.SPREAD and cpos > -self.pos_limits[product]):
                buy_amount = min(-amount, self.pos_limits[product] - cpos)
                if buy_amount > 0:
                    print("BUY", str(buy_amount) + "x", price)
                    cpos += buy_amount
                    total_buy_amount += buy_amount
                    orders.append(Order(product, price, buy_amount))
        

        if isinstance(state.position.get('ORCHIDS'), int) and total_buy_amount != 0:
            self.orchids_conversions = state.position.get('ORCHIDS') - total_buy_amount

                
        return orders, self.orchids_conversions
    
    
    
    def trade_basket(self, state: TradingState):
        orders = {'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        
        min_bids, max_bids, bid_volumes, best_bid_volumes, min_asks, max_asks, ask_volumes, best_ask_volumes, best_bid_prices, best_ask_prices, buy_depths, ask_depths = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        orders_sell, orders_buy = {}, {}
        mid_prices = {}
        # Initialize Data
        for product in prods:
            
            if state.order_depths.get(product) == None:
                continue
            order_sell, order_buy = collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items())), collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True))
            orders_sell[product], orders_buy[product] = order_sell, order_buy
            min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, buy_depth, ask_depth = self.extract_values(order_sell, order_buy)
            min_bids[product], max_bids[product], bid_volumes[product], best_bid_volumes[product], min_asks[product], max_asks[product], ask_volumes[product], best_ask_volumes[product], best_bid_prices[product], best_ask_prices[product], buy_depths[product], ask_depths[product] = min_bid, max_bid, bid_volume, best_bid_volume, min_ask, max_ask, ask_volume, best_ask_volume, best_bid_price, best_ask_price, buy_depth, ask_depth
            
            mid_prices[product] = statistics.median([max_bid, min_ask])
            self.add_to_cache(product.lower(), mid_prices[product])
    

        excess = mid_prices['GIFT_BASKET'] - 4*mid_prices['CHOCOLATE'] - 6*mid_prices['STRAWBERRIES'] - mid_prices['ROSES'] - 379
        ideal_price = int(mid_prices['GIFT_BASKET'] - excess)
        print("Excess: " + str(excess))
        
        # if excess is positive, we sell gift baskets, buy chocolates and strawberries and roses
        if abs(excess) >= 76 and excess > 0:
            # Sell gift baskets
            basket_pos = self.cpos['GIFT_BASKET']
            for price, amount in orders_buy['GIFT_BASKET'].items():
                if price >= ideal_price + self.REGRESSION_SPREAD and basket_pos > -self.pos_limits['GIFT_BASKET']:
                    sell_amount = max(-amount, -self.pos_limits['GIFT_BASKET'] - basket_pos)
                    if sell_amount < 0:
                        print("SELL", str(sell_amount) + "x", price)
                        orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, sell_amount))
                        basket_pos += sell_amount   
                        
            # Penny Sell
            if basket_pos > -self.pos_limits['GIFT_BASKET']:
                num = -self.pos_limits['GIFT_BASKET'] - basket_pos
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', ideal_price + 4, num))
                basket_pos += num
                        
            
            # Buying the rest checking if they are currently under price, using SMA5  
            target_price_choc = self.sma_price('CHOCOLATE') - self.SPREAD                   
            
            choc_pos = self.cpos['CHOCOLATE']
            for price, amount in orders_sell['CHOCOLATE'].items():
                if price <= target_price_choc and choc_pos < self.pos_limits['CHOCOLATE']:
                    buy_amount = min(amount, self.pos_limits['CHOCOLATE'] - choc_pos)
                    if buy_amount > 0:
                        print("BUY", str(buy_amount) + "x", price)
                        orders['CHOCOLATE'].append(Order('CHOCOLATE', price, buy_amount))
                        choc_pos += buy_amount
            
            # Penny
            if choc_pos < self.pos_limits['CHOCOLATE']:
                num = self.pos_limits['CHOCOLATE'] - choc_pos
                orders['CHOCOLATE'].append(Order('CHOCOLATE', target_price_choc - 4, num))
                choc_pos += num
                
                
            target_price_straw = self.sma_price('STRAWBERRIES') - self.SPREAD
            straw_pos = self.cpos['STRAWBERRIES']
            for price, amount in orders_sell['STRAWBERRIES'].items():
                if price <= target_price_straw and straw_pos < self.pos_limits['STRAWBERRIES']:
                    buy_amount = min(amount, self.pos_limits['STRAWBERRIES'] - straw_pos)
                    if buy_amount > 0:
                        print("BUY", str(buy_amount) + "x", price)
                        orders['STRAWBERRIES'].append(Order('STRAWBERRIES', price, buy_amount))
                        straw_pos += buy_amount
                        
            # Penny
            if straw_pos < self.pos_limits['STRAWBERRIES']:
                num = self.pos_limits['STRAWBERRIES'] - straw_pos
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', target_price_straw - 4, num))
                straw_pos += num
                
            target_price_rose = self.sma_price('ROSES') - self.SPREAD
            rose_pos = self.cpos['ROSES']
            for price, amount in orders_sell['ROSES'].items():
                if price <= target_price_rose and rose_pos < self.pos_limits['ROSES']:
                    buy_amount = min(amount, self.pos_limits['ROSES'] - rose_pos)
                    if buy_amount > 0:
                        print("BUY", str(buy_amount) + "x", price)
                        orders['ROSES'].append(Order('ROSES', price, buy_amount))
                        rose_pos += buy_amount
        
        elif abs(excess) >= 76 and excess < 0:
            # if excess is negative, we buy gift baskets, sell chocolates and strawberries and roses
            basket_pos = self.cpos['GIFT_BASKET']
            for price, amount in orders_sell['GIFT_BASKET'].items():
                if price <= ideal_price - self.SPREAD and basket_pos < self.pos_limits['GIFT_BASKET']:
                    buy_amount = min(amount, self.pos_limits['GIFT_BASKET'] - basket_pos)
                    if buy_amount > 0:
                        print("BUY", str(buy_amount) + "x", price)
                        orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, buy_amount))
                        basket_pos += buy_amount
                        
            # Penny
            if basket_pos < self.pos_limits['GIFT_BASKET']:
                num = self.pos_limits['GIFT_BASKET'] - basket_pos
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', ideal_price - 4, num))
                basket_pos += num
            
            # Selling the rest
            target_price_choc = self.sma_price('CHOCOLATE') + self.SPREAD
            choc_pos = self.cpos['CHOCOLATE']
            for price, amount in orders_buy['CHOCOLATE'].items():
                if price >= target_price_choc and choc_pos > -self.pos_limits['CHOCOLATE']:
                    sell_amount = max(-amount, -self.pos_limits['CHOCOLATE'] - choc_pos)
                    if sell_amount < 0:
                        print("SELL", str(sell_amount) + "x", price)
                        orders['CHOCOLATE'].append(Order('CHOCOLATE', price, sell_amount))
                        choc_pos += sell_amount
            
            # Penny
            if choc_pos > -self.pos_limits['CHOCOLATE']:
                num = -self.pos_limits['CHOCOLATE'] - choc_pos
                orders['CHOCOLATE'].append(Order('CHOCOLATE', target_price_choc + 4, num))
                choc_pos += num
                
            target_price_straw = self.sma_price('STRAWBERRIES') + self.SPREAD
            straw_pos = self.cpos['STRAWBERRIES']
            for price, amount in orders_buy['STRAWBERRIES'].items():
                if price >= target_price_straw and straw_pos > -self.pos_limits['STRAWBERRIES']:
                    sell_amount = max(-amount, -self.pos_limits['STRAWBERRIES'] - straw_pos)
                    if sell_amount < 0:
                        print("SELL", str(sell_amount) + "x", price)
                        orders['STRAWBERRIES'].append(Order('STRAWBERRIES', price, sell_amount))
                        straw_pos += sell_amount
            
            # Penny
            if straw_pos > -self.pos_limits['STRAWBERRIES']:
                num = -self.pos_limits['STRAWBERRIES'] - straw_pos
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', target_price_straw + 4, num))
                straw_pos += num
                
            target_price_rose = self.sma_price('ROSES') + self.SPREAD
            rose_pos = self.cpos['ROSES']
            for price, amount in orders_buy['ROSES'].items():
                if price >= target_price_rose and rose_pos > -self.pos_limits['ROSES']:
                    sell_amount = max(-amount, -self.pos_limits['ROSES'] - rose_pos)
                    if sell_amount < 0:
                        print("SELL", str(sell_amount) + "x", price)
                        orders['ROSES'].append(Order('ROSES', price, sell_amount))
                        rose_pos += sell_amount
            
            # Penny
            if rose_pos > -self.pos_limits['ROSES']:
                num = -self.pos_limits['ROSES'] - rose_pos
                orders['ROSES'].append(Order('ROSES', target_price_rose + 4, num))
                rose_pos += num
                
                
        else:
            orders['GIFT_BASKET'] = self.market_make_regression('GIFT_BASKET', ideal_price, orders_sell['GIFT_BASKET'], orders_buy['GIFT_BASKET'], self.cpos['GIFT_BASKET'])
            orders['CHOCOLATE'] = self.market_make_regression('CHOCOLATE', self.sma_price('CHOCOLATE'), orders_sell['CHOCOLATE'], orders_buy['CHOCOLATE'], self.cpos['CHOCOLATE'])
            orders['STRAWBERRIES'] = self.market_make_regression('STRAWBERRIES', self.sma_price('STRAWBERRIES'), orders_sell['STRAWBERRIES'], orders_buy['STRAWBERRIES'], self.cpos['STRAWBERRIES'])
            orders['ROSES'] = self.market_make_regression('ROSES', self.sma_price('ROSES'), orders_sell['ROSES'], orders_buy['ROSES'], self.cpos['ROSES'])
                
        return orders  
    
    
    def add_to_cache(self, product, data):
        attr_name = f'{product.lower()}_cache'
        cache = getattr(self, attr_name)
        
        if len(cache) < 5:
            cache.append(data)
        else:
            cache.append(data)
            cache.pop(0)
            
    def sma_price(self, product):
        attr_name = f'{product.lower()}_cache'
        cache = getattr(self, attr_name)
        return int(sum(cache) / len(cache))
    
    
    def market_make_regression(self, product, target_price, orders_sell, orders_buy, curr_pos):
        orders = []
        
        cpos = curr_pos
        
        for price, amount in orders_buy.items():
            # Current price is greater than predicted next price, so we sell, amount here is positive
            if (price >= target_price + self.REGRESSION_SPREAD and cpos > -self.pos_limits[product]):
                #or (cpos > 0 and price+1 == target_price + self.REGRESSION_SPREAD)):
              #if (price <= target_price - self.SPREAD and cpos > -self.pos_limits[product]):
                sell_amount = max(-amount, -self.pos_limits[product] - cpos)
            
                if sell_amount < 0:
                    print("SELL", str(sell_amount) + "x", price)
                    cpos += sell_amount
                    orders.append(Order(product, price, sell_amount))

        if (cpos > -self.pos_limits[product]):
            num = -self.pos_limits[product]-cpos
            orders.append(Order(product, target_price + self.SPREAD, num))
            cpos += num


        cpos = curr_pos
        
        for price, amount in orders_sell.items():
            # Current lower than predicted next price, so we buy, amount here is negative
            if (price <= target_price - self.REGRESSION_SPREAD and cpos < self.pos_limits[product]):
                #or (cpos < 0 and price-1 == target_price - self.REGRESSION_SPREAD)):
            #if (price >= target_price + self.SPREAD and cpos < self.pos_limits[product]):
                buy_amount = min(-amount, self.pos_limits[product] - cpos)
                if buy_amount > 0:
                    print("BUY", str(buy_amount) + "x", price)
                    orders.append(Order(product, price, buy_amount))
                    cpos += buy_amount
        
                
        if cpos < self.pos_limits[product]:
            num = self.pos_limits[product] - cpos
            orders.append(Order(product, target_price - self.SPREAD, num))
            cpos += num
            
        return orders
    

    
    
    
    
                
                
    

    
    
    
    
        

            
        
        
                                                        
                                                        
            
        
        
        
            
        
        
            
    
    
        
            
            
    
            
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        

        
    
    
    
    
    
    
