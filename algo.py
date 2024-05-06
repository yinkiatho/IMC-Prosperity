import json
from typing import Any, Dict, List
import numpy as np
import math
import statistics
import jsonpickle
import pandas as pd
import collections
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
    std_perc = 0.5
    
    
    def __init__(self, linear_regression=[6, [0.0, 0.9487079673973507, 0.04882953537331608, 0.0, 0.001374535182263223, 0.0], 5.475369188194236],
                        SPREAD=3, REGRESSION_SPREAD=1, extra=20, regression_extra=20, lower_bound=0, upper_bound=15, ORCHIDS_LIMIT=200,
                        price_diff=3, days=3, std_perc = 0.5):
        
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
        self.std_perc = std_perc
        
    
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
        ##print(self.starfruit_cache)
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
                        #print("SELL", str(sell_amount) + "x", price)
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
                        #print("BUY", str(buy_amount) + "x", price)
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
                    #print("SELL", str(sell_amount) + "x", price)
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
                    #print("BUY", str(buy_amount) + "x", price)
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
                    #print("SELL", str(sell_amount) + "x", price)
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
                    #print("BUY", str(buy_amount) + "x", price)
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
    

        ideal_price = 4*mid_prices['CHOCOLATE'] + 6*mid_prices['STRAWBERRIES'] + mid_prices['ROSES'] + 379
        excess = mid_prices['GIFT_BASKET'] - ideal_price
        #print("Excess: " + str(excess))
        
        # if excess is positive, we sell gift baskets, buy chocolates and strawberries and roses
        if excess >= 76*0.5:
            #print("Arbitrage Opportunity: Buying Chocolates, Strawberries and Roses and Selling Gift Baskets")
            TARGET = best_bid_prices['GIFT_BASKET'] - 1
            # Sell gift baskets
            basket_pos = self.cpos['GIFT_BASKET']
            for price, amount in orders_buy['GIFT_BASKET'].items():
                if price >= TARGET and basket_pos > -self.pos_limits['GIFT_BASKET']:
                    sell_amount = max(-amount, -self.pos_limits['GIFT_BASKET'] - basket_pos)
                    if sell_amount < 0:
                        #print("SELL", str(sell_amount) + "x", price)
                        orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, sell_amount))
                        basket_pos += sell_amount   

            # Penny Sell
            if basket_pos > -self.pos_limits['GIFT_BASKET']:
                num = -self.pos_limits['GIFT_BASKET'] - basket_pos
                #orders['GIFT_BASKET'].append(Order('GIFT_BASKET', ideal_price + 4, num))
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', TARGET, num))
                basket_pos += num

                        
            
            # Buying the rest checking if they are currently under price
            orders['CHOCOLATE'] = self.market_make_momentum('CHOCOLATE', best_bid_price=best_bid_prices['CHOCOLATE'], 
                                                            best_ask_price=best_ask_prices['CHOCOLATE'], 
                                                            orders_sell=orders_sell['CHOCOLATE'], 
                                                            orders_buy=orders_buy['CHOCOLATE'], force_buy=True)
            
            orders['STRAWBERRIES'] = self.market_make_momentum('STRAWBERRIES', best_bid_price=best_bid_prices['STRAWBERRIES'],
                                                            best_ask_price=best_ask_prices['STRAWBERRIES'],
                                                            orders_sell=orders_sell['STRAWBERRIES'],
                                                            orders_buy=orders_buy['STRAWBERRIES'], force_buy=True)
            orders['ROSES'] = self.market_make_momentum('ROSES', best_bid_price=best_bid_prices['ROSES'],
                                                            best_ask_price=best_ask_prices['ROSES'],
                                                            orders_sell=orders_sell['ROSES'],
                                                            orders_buy=orders_buy['ROSES'], force_buy=True)
            
        
        elif excess < -76*self.std_perc:
            #print("Arbitrage Opportunity: Buying Gift Baskets and Selling Chocolates, Strawberries and Roses")
            # if excess is negative, we buy gift baskets, sell chocolates and strawberries and roses
            basket_pos = self.cpos['GIFT_BASKET']
            TARGET = best_ask_prices['GIFT_BASKET'] + 1
            for price, amount in orders_sell['GIFT_BASKET'].items():
                if price <= TARGET and basket_pos < self.pos_limits['GIFT_BASKET']:
                    buy_amount = min(amount, self.pos_limits['GIFT_BASKET'] - basket_pos)
                    if buy_amount > 0:
                        #print("BUY", str(buy_amount) + "x", price)
                        orders['GIFT_BASKET'].append(Order('GIFT_BASKET', price, buy_amount))
                        basket_pos += buy_amount
                        
            # Penny
            if basket_pos < self.pos_limits['GIFT_BASKET']:
                num = self.pos_limits['GIFT_BASKET'] - basket_pos
                #orders['GIFT_BASKET'].append(Order('GIFT_BASKET', ideal_price - 4, num))
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', TARGET, num))
                basket_pos += num
            
            # Selling the rest
            orders['CHOCOLATE'] = self.market_make_momentum('CHOCOLATE', best_bid_price=best_bid_prices['CHOCOLATE'], 
                                                            best_ask_price=best_ask_prices['CHOCOLATE'], 
                                                            orders_sell=orders_sell['CHOCOLATE'], 
                                                            orders_buy=orders_buy['CHOCOLATE'], force_sell=True)
            
            orders['STRAWBERRIES'] = self.market_make_momentum('STRAWBERRIES', best_bid_price=best_bid_prices['STRAWBERRIES'],
                                                            best_ask_price=best_ask_prices['STRAWBERRIES'],
                                                            orders_sell=orders_sell['STRAWBERRIES'],
                                                            orders_buy=orders_buy['STRAWBERRIES'], force_sell=True)
            
            orders['ROSES'] = self.market_make_momentum('ROSES', best_bid_price=best_bid_prices['ROSES'],
                                                            best_ask_price=best_ask_prices['ROSES'],
                                                            orders_sell=orders_sell['ROSES'],
                                                            orders_buy=orders_buy['ROSES'], force_sell=True)
                
        # elif abs(excess) < 76*self.std_perc*0.5:
        #     # We are within the standard deviation, exiting positions
        #     for product in prods:
        #         orders[product] = self.exit_positions(product, best_bid_prices[product], best_ask_prices[product])     
        return orders  
    
    
    def exit_positions(self, product, best_bid_price, best_ask_price):
        cpos = self.cpos[product]
        orders = []
        #price = self.ewma_price(product, 5)
        #price  = self.sma_price(product, 5)
        # We want to sell our long position
        if cpos > 0:
            orders.append(Order(product, best_ask_price-1, -cpos))
            #orders.append(Order(product, best_bid_price - 1, -cpos))
            #orders.append(Order(product, price + 1, -cpos))
        elif cpos < 0:
            orders.append(Order(product, best_bid_price+1, -cpos))
            #orders.append(Order(product, best_ask_price + 1, cpos))
            #orders.append(Order(product, price - 1, cpos))
            
        return orders
    
    
    def add_to_cache(self, product, data):
        attr_name = f'{product.lower()}_cache'
        cache = getattr(self, attr_name)
        
        if len(cache) < 10:
            cache.append(data)
        else:
            cache.append(data)
            cache.pop(0)
            
    def sma_price(self, product, n):
        attr_name = f'{product.lower()}_cache'
        cache = getattr(self, attr_name)
        return int(sum(cache[-n:]) / n)
    
    def ewma_price(self, product, n):
        attr_name = f'{product.lower()}_cache'
        cache = getattr(self, attr_name)
        return int(sum([(2**(i+1)) * cache[-(i+1)] for i in range(n)]) / sum([2**(i+1) for i in range(n)]))
    

    def check_momentum(self, product):
        sma5, sma10 = self.sma_price(product, 5), self.sma_price(product, 10)
        if sma5 > sma10:
            return 1
        else:
            return -1
        
    
    def market_make_momentum(self, product, best_bid_price, best_ask_price, orders_sell, orders_buy, force_buy=False, force_sell=False):
        orders = []
        cpos = self.cpos[product]
        curr_amount = 0
        if (self.check_momentum(product) == -1 and force_buy == False) or (force_buy == True and force_sell == False):
            for price, amount in orders_sell.items():
                if price <= best_bid_price+1 and cpos < self.pos_limits[product]:
                    buy_amount = min(amount, self.pos_limits[product] - cpos)
                    if buy_amount > 0:
                        #print("BUY", str(buy_amount) + "x", price)
                        orders.append(Order(product, price, buy_amount))
                        cpos += buy_amount
                        curr_amount += buy_amount
                        
            if cpos < self.pos_limits[product]:
                num = self.pos_limits[product] - cpos
                orders.append(Order(product, best_bid_price+1, num))
                cpos += num
                
        elif self.check_momentum(product) == 1 and force_sell == False or (force_buy == False and force_sell == True):
            for price, amount in orders_buy.items():
                if price >= best_ask_price-1 and cpos > -self.pos_limits[product]:
                    sell_amount = max(-amount, -self.pos_limits[product] - cpos)
                    if sell_amount < 0:
                        #print("SELL", str(sell_amount) + "x", price)
                        orders.append(Order(product, price, sell_amount))
                        cpos += sell_amount
                        curr_amount += sell_amount
                        
            if cpos > -self.pos_limits[product]:
                num = -self.pos_limits[product] - cpos
                orders.append(Order(product, best_ask_price-1, num))
                cpos += num
        return orders
        
    

    
    
    
    
                
                
    

    
    
    
    
        

            
        
        
                                                        
                                                        
            
        
        
        
            
        
        
            
    
    
        
            
            
    
            
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        

        
    
    
    
    
    
    
