from dontlooseshells_algo import Trader
from datamodel import *
from typing import Any  #, Callable
import numpy as np
import pandas as pd
import statistics
import copy
import uuid
import random
import os
from datetime import datetime

# Timesteps used in training files
TIME_DELTA = 100
# Please put all! the price and log files into
# the same directory or adjust the code accordingly
TRAINING_DATA_PREFIX = "./training"

ALL_SYMBOLS = [
    'PEARLS',
    'BANANAS',
    'COCONUTS',
    'PINA_COLADAS',
    'DIVING_GEAR',
    'BERRIES',
    'DOLPHIN_SIGHTINGS',
    'BAGUETTE',
    'DIP',
    'UKULELE',
    'PICNIC_BASKET',
    'AMETHYSTS',
    'STARFRUIT'
]
POSITIONABLE_SYMBOLS = [
    'PEARLS',
    'BANANAS',
    'COCONUTS',
    'PINA_COLADAS',
    'DIVING_GEAR',
    'BERRIES',
    'BAGUETTE',
    'DIP',
    'UKULELE',
    'PICNIC_BASKET',
    'AMETHYSTS',
    'STARFRUIT'
]
first_round = ['AMETHYSTS', 'STARFRUIT']
snd_round = first_round + ['COCONUTS',  'PINA_COLADAS']
third_round = snd_round + ['DIVING_GEAR', 'DOLPHIN_SIGHTINGS', 'BERRIES']
fourth_round = third_round + ['BAGUETTE', 'DIP', 'UKULELE', 'PICNIC_BASKET']
fifth_round = fourth_round # + secret, maybe pirate gold?

SYMBOLS_BY_ROUND = {
    1: first_round,
    2: snd_round,
    3: third_round,
    4: fourth_round,
    5: fifth_round,
}

first_round_pst =['AMETHYSTS', 'STARFRUIT']
snd_round_pst = first_round_pst + ['COCONUTS',  'PINA_COLADAS']
third_round_pst = snd_round_pst + ['DIVING_GEAR', 'BERRIES']
fourth_round_pst = third_round_pst + ['BAGUETTE', 'DIP', 'UKULELE', 'PICNIC_BASKET']
fifth_round_pst = fourth_round_pst # + secret, maybe pirate gold?

SYMBOLS_BY_ROUND_POSITIONABLE = {
    1: first_round_pst,
    2: snd_round_pst,
    3: third_round_pst,
    4: fourth_round_pst,
    5: fifth_round_pst,
}

def process_prices(df_prices, round, time_limit) -> dict[int, TradingState]:
    states = {}
    for _, row in df_prices.iterrows():
        time: int = int(row["timestamp"])
        if time > time_limit:
            break
        product: str = row["product"]
        if states.get(time) == None:
            position: Dict[Product, Position] = {}
            own_trades: Dict[Symbol, List[Trade]] = {}
            market_trades: Dict[Symbol, List[Trade]] = {}
            observations: Dict[Product, Observation] = {}
            listings = {}
            depths = {}
            states[time] = TradingState("start" , time , listings, depths, own_trades, market_trades, position, observations)

        if product not in states[time].position and product in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
            states[time].position[product] = 0
            states[time].own_trades[product] = []
            states[time].market_trades[product] = []

        states[time].listings[product] = Listing(product, product, "1")

        if product == "DOLPHIN_SIGHTINGS":
            states[time].observations["DOLPHIN_SIGHTINGS"] = row['mid_price']
            
        depth = OrderDepth()
        if row["bid_price_1"]> 0:
            depth.buy_orders[row["bid_price_1"]] = int(row["bid_volume_1"])
        if row["bid_price_2"]> 0:
            depth.buy_orders[row["bid_price_2"]] = int(row["bid_volume_2"])
        if row["bid_price_3"]> 0:
            depth.buy_orders[row["bid_price_3"]] = int(row["bid_volume_3"])
        if row["ask_price_1"]> 0:
            depth.sell_orders[row["ask_price_1"]] = -int(row["ask_volume_1"])
        if row["ask_price_2"]> 0:
            depth.sell_orders[row["ask_price_2"]] = -int(row["ask_volume_2"])
        if row["ask_price_3"]> 0:
            depth.sell_orders[row["ask_price_3"]] = -int(row["ask_volume_3"])
        states[time].order_depths[product] = depth

    return states

def process_trades(df_trades, states: dict[int, TradingState], time_limit, names=True):
    for _, trade in df_trades.iterrows():
        time: int = trade['timestamp']
        if time > time_limit:
            break
        symbol = trade['symbol']
        if symbol not in states[time].market_trades:
            states[time].market_trades[symbol] = []
        t = Trade(
                symbol, 
                trade['price'], 
                trade['quantity'], 
                str(trade['buyer']), 
                str(trade['seller']),
                time)
        states[time].market_trades[symbol].append(t)
    return states
       
current_limits = {
    'PEARLS': 20,
    'BANANAS': 20,
    'COCONUTS': 600,
    'PINA_COLADAS': 300,
    'DIVING_GEAR': 50,
    'BERRIES': 250,
    'BAGUETTE': 150,
    'DIP': 300,
    'UKULELE': 70,
    'PICNIC_BASKET': 70,
    'AMETHYSTS': 20,
    'STARFRUIT': 20,
}

def calc_mid(states: dict[int, TradingState], round: int, time: int, max_time: int) -> dict[str, float]:
    medians_by_symbol = {}
    non_empty_time = time
    all_mids = {}
    for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
        hitted_zero = False
        while len(states[non_empty_time].order_depths[psymbol].sell_orders.keys()) == 0 or len(states[non_empty_time].order_depths[psymbol].buy_orders.keys()) == 0:
            # little hack
            if time == 0 or hitted_zero and time != max_time:
                hitted_zero = True
                non_empty_time += TIME_DELTA
            else:
                non_empty_time -= TIME_DELTA
        min_ask = min(states[non_empty_time].order_depths[psymbol].sell_orders.keys())
        max_bid = max(states[non_empty_time].order_depths[psymbol].buy_orders.keys())
        median_price = statistics.median([min_ask, max_bid])
        medians_by_symbol[psymbol] = median_price
        
    
    return medians_by_symbol


def calculate_mid_prices(states: dict[int, TradingState], max_time: int):
    all_mids = {}
    for symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
        all_mids[symbol] = []
        for time, state in states.items():
            min_ask = min(state.order_depths[symbol].sell_orders.keys())
            max_bid = max(state.order_depths[symbol].buy_orders.keys())
            median_price = statistics.median([min_ask, max_bid])
            all_mids[symbol].append(median_price)
            
    return all_mids
            
            
    


# Setting a high time_limit can be harder to visualize
# print_position prints the position before! every Trader.run
def simulate_alternative(
        round: int, 
        day: int, 
        trader, 
        time_limit=999900, 
        names=True, 
        halfway=False,
        monkeys=False,
        monkey_names=['Caesar', 'Camilla', 'Peter']
    ):
    prices_path = os.path.join(TRAINING_DATA_PREFIX, f'prices_round_{round}_day_{day}.csv')
    trades_path = os.path.join(TRAINING_DATA_PREFIX, f'trades_round_{round}_day_{day}_nn.csv')
    if not names:
        trades_path = os.path.join(TRAINING_DATA_PREFIX, f'trades_round_{round}_day_{day}_nn.csv')
    df_prices = pd.read_csv(prices_path, sep=';')
    df_trades = pd.read_csv(trades_path, sep=';', dtype={ 'seller': str, 'buyer': str })

    states = process_prices(df_prices, round, time_limit)
    states = process_trades(df_trades, states, time_limit, names)
    ref_symbols = list(states[0].position.keys())
    max_time = max(list(states.keys()))
    
    all_mids = calculate_mid_prices(states, max_time)

    # handling these four is rather tricky 
    profits_by_symbol: dict[int, dict[str, float]] = { 0: dict(zip(ref_symbols, [0.0]*len(ref_symbols))) }
    balance_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }
    credit_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }
    unrealized_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }

    states, trader, profits_by_symbol, balance_by_symbol = trades_position_pnl_run(states, max_time, profits_by_symbol, balance_by_symbol, credit_by_symbol, unrealized_by_symbol, trader)
    create_log_file(round, day, states, profits_by_symbol, balance_by_symbol, trader)
    profit_balance_monkeys = {}
    trades_monkeys = {}
    if monkeys:
        profit_balance_monkeys, trades_monkeys, profit_monkeys, balance_monkeys, monkey_positions_by_timestamp = monkey_positions(monkey_names, states, round)
        print("End of monkey simulation reached.")
        print(f'PNL + BALANCE monkeys {profit_balance_monkeys[max_time]}')
        print(f'Trades monkeys {trades_monkeys[max_time]}')
    if hasattr(trader, 'after_last_round'):
        if callable(trader.after_last_round): #type: ignore
            trader.after_last_round(profits_by_symbol, balance_by_symbol) #type: ignore
            
    return profits_by_symbol, all_mids


def trades_position_pnl_run(
        states: dict[int, TradingState],
        max_time: int, 
        profits_by_symbol: dict[int, dict[str, float]], 
        balance_by_symbol: dict[int, dict[str, float]], 
        credit_by_symbol: dict[int, dict[str, float]], 
        unrealized_by_symbol: dict[int, dict[str, float]], 
        trader):
        starting = True
        traderState = None
        #mids, all_mids = calc_mid(states, round, time, max_time)
        for time, state in states.items():
            position = copy.deepcopy(state.position)
            if not starting and traderState != None:
                state.traderData = traderState
                
            orders, conversions, traderData = trader.run(state)
            if starting:
                traderState = traderData
                starting = False
                
            if not starting:
                traderState = traderData
            mids = calc_mid(states, round, time, max_time)
            trades = clear_order_book(orders, state.order_depths, time, halfway)
            if profits_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                profits_by_symbol[time + TIME_DELTA] = copy.deepcopy(profits_by_symbol[time])
            if credit_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                credit_by_symbol[time + TIME_DELTA] = copy.deepcopy(credit_by_symbol[time])
            if balance_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                balance_by_symbol[time + TIME_DELTA] = copy.deepcopy(balance_by_symbol[time])
            if unrealized_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                unrealized_by_symbol[time + TIME_DELTA] = copy.deepcopy(unrealized_by_symbol[time])
                for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                    unrealized_by_symbol[time + TIME_DELTA][psymbol] = mids[psymbol]*position[psymbol]
            valid_trades = []
            failed_symbol = []
            grouped_by_symbol = {}
            if len(trades) > 0:
                for trade in trades:
                    if trade.symbol in failed_symbol:
                        continue
                    n_position = position[trade.symbol] + trade.quantity
                    if abs(n_position) > current_limits[trade.symbol]:
                        print('ILLEGAL TRADE, WOULD EXCEED POSITION LIMIT, KILLING ALL REMAINING ORDERS')
                        trade_vars = vars(trade)
                        trade_str = ', '.join("%s: %s" % item for item in trade_vars.items())
                        print(f'Stopped at the following trade: {trade_str}')
                        print(f"All trades that were sent:")
                        for trade in trades:
                            trade_vars = vars(trade)
                            trades_str = ', '.join("%s: %s" % item for item in trade_vars.items())
                            print(trades_str)
                        failed_symbol.append(trade.symbol)
                    else:
                        valid_trades.append(trade) 
                        position[trade.symbol] += trade.quantity
            FLEX_TIME_DELTA = TIME_DELTA
            if time == max_time:
                FLEX_TIME_DELTA = 0
            for valid_trade in valid_trades:
                    if grouped_by_symbol.get(valid_trade.symbol) == None:
                        grouped_by_symbol[valid_trade.symbol] = []
                    grouped_by_symbol[valid_trade.symbol].append(valid_trade)
                    credit_by_symbol[time + FLEX_TIME_DELTA][valid_trade.symbol] += -valid_trade.price * valid_trade.quantity
            if states.get(time + FLEX_TIME_DELTA) != None:
                states[time + FLEX_TIME_DELTA].own_trades = grouped_by_symbol
                for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                    unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol] = mids[psymbol]*position[psymbol]
                    if position[psymbol] == 0 and states[time].position[psymbol] != 0:
                        profits_by_symbol[time + FLEX_TIME_DELTA][psymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] #+unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol]
                        credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] = 0
                        balance_by_symbol[time + FLEX_TIME_DELTA][psymbol] = 0
                    else:
                        balance_by_symbol[time + FLEX_TIME_DELTA][psymbol] = credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol]

            if time == max_time:
                print("End of simulation reached. All positions left are liquidated")
                # i have the feeling this already has been done, and only repeats the same values as before
                for osymbol in position.keys():
                    profits_by_symbol[time + FLEX_TIME_DELTA][osymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][osymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][osymbol]
                    balance_by_symbol[time + FLEX_TIME_DELTA][osymbol] = 0
            if states.get(time + FLEX_TIME_DELTA) != None:
                states[time + FLEX_TIME_DELTA].position = copy.deepcopy(position)
        return states, trader, profits_by_symbol, balance_by_symbol

def monkey_positions(monkey_names: list[str], states: dict[int, TradingState], round):
    profits_by_symbol: dict[int, dict[str, dict[str, float]]] = { 0: {} }
    balance_by_symbol: dict[int, dict[str, dict[str, float]]] =  { 0: {} }
    credit_by_symbol: dict[int, dict[str, dict[str, float]]] = { 0: {} }
    unrealized_by_symbol: dict[int, dict[str, dict[str, float]]] = { 0: {} }
    prev_monkey_positions: dict[str, dict[str, int]] = {}
    monkey_positions: dict[str, dict[str, int]] = {}
    trades_by_round: dict[int, dict[str, list[Trade]]]  = { 0: dict(zip(monkey_names,  [[] for x in range(len(monkey_names))])) }
    profit_balance: dict[int, dict[str, dict[str, float]]] = { 0: {} }

    monkey_positions_by_timestamp: dict[int, dict[str, dict[str, int]]] = {}

    for monkey in monkey_names:
        ref_symbols = list(states[0].position.keys())
        profits_by_symbol[0][monkey] = dict(zip(ref_symbols, [0.0]*len(ref_symbols)))
        balance_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        credit_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        unrealized_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        profit_balance[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        monkey_positions[monkey] = dict(zip(SYMBOLS_BY_ROUND_POSITIONABLE[round], [0]*len(SYMBOLS_BY_ROUND_POSITIONABLE[round])))
        prev_monkey_positions[monkey] = copy.deepcopy(monkey_positions[monkey])

    for time, state in states.items():
        already_calculated = False
        for monkey in monkey_names:
            position = copy.deepcopy(monkey_positions[monkey])
            mids = calc_mid(states, round, time, max_time)
            if trades_by_round.get(time + TIME_DELTA) == None:
                trades_by_round[time + TIME_DELTA] =  copy.deepcopy(trades_by_round[time])

            for psymbol in POSITIONABLE_SYMBOLS:
                if already_calculated:
                    break
                if state.market_trades.get(psymbol):
                    for market_trade in state.market_trades[psymbol]:
                        if trades_by_round[time].get(market_trade.buyer) != None:
                            trades_by_round[time][market_trade.buyer].append(Trade(psymbol, market_trade.price, market_trade.quantity))
                        if trades_by_round[time].get(market_trade.seller) != None:
                            trades_by_round[time][market_trade.seller].append(Trade(psymbol, market_trade.price, -market_trade.quantity))
            already_calculated = True

            if profit_balance.get(time + TIME_DELTA) == None and time != max_time:
                profit_balance[time + TIME_DELTA] = copy.deepcopy(profit_balance[time])
            if profits_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                profits_by_symbol[time + TIME_DELTA] = copy.deepcopy(profits_by_symbol[time])
            if credit_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                credit_by_symbol[time + TIME_DELTA] = copy.deepcopy(credit_by_symbol[time])
            if balance_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                balance_by_symbol[time + TIME_DELTA] = copy.deepcopy(balance_by_symbol[time])
            if unrealized_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                unrealized_by_symbol[time + TIME_DELTA] = copy.deepcopy(unrealized_by_symbol[time])
                for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                    unrealized_by_symbol[time + TIME_DELTA][monkey][psymbol] = mids[psymbol]*position[psymbol]
            valid_trades = []
            if trades_by_round[time].get(monkey) != None:  
                valid_trades = trades_by_round[time][monkey]
            FLEX_TIME_DELTA = TIME_DELTA
            if time == max_time:
                FLEX_TIME_DELTA = 0
            for valid_trade in valid_trades:
                    position[valid_trade.symbol] += valid_trade.quantity
                    credit_by_symbol[time + FLEX_TIME_DELTA][monkey][valid_trade.symbol] += -valid_trade.price * valid_trade.quantity
            if states.get(time + FLEX_TIME_DELTA) != None:
                for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                    unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] = mids[psymbol]*position[psymbol]
                    if position[psymbol] == 0 and prev_monkey_positions[monkey][psymbol] != 0:
                        profits_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol]
                        credit_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] = 0
                        balance_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] = 0
                    else:
                        balance_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] = credit_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol]
            profit_balance[time + FLEX_TIME_DELTA][monkey][psymbol] = profits_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] + balance_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol]
            prev_monkey_positions[monkey] = copy.deepcopy(monkey_positions[monkey])
            monkey_positions[monkey] = position
            if time == max_time:
                # i have the feeling this already has been done, and only repeats the same values as before
                for osymbol in position.keys():
                    profits_by_symbol[time + FLEX_TIME_DELTA][monkey][osymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][monkey][osymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][osymbol]
                    balance_by_symbol[time + FLEX_TIME_DELTA][monkey][osymbol] = 0
        monkey_positions_by_timestamp[time] = copy.deepcopy(monkey_positions)
    return profit_balance, trades_by_round, profits_by_symbol, balance_by_symbol, monkey_positions_by_timestamp


def cleanup_order_volumes(org_orders: List[Order]) -> List[Order]:
    orders = []
    for order_1 in org_orders:
        final_order = copy.copy(order_1)
        for order_2 in org_orders:
            if order_1.price == order_2.price and order_1.quantity == order_2.quantity:
               continue 
            if order_1.price == order_2.price:
                final_order.quantity += order_2.quantity
        orders.append(final_order)
    return orders

def clear_order_book(trader_orders: dict[str, List[Order]], order_depth: dict[str, OrderDepth], time: int, halfway: bool) -> list[Trade]:
        trades = []
        print(trader_orders)
        for symbol in trader_orders.keys():
            if order_depth.get(symbol) != None:
                symbol_order_depth = copy.deepcopy(order_depth[symbol])
                t_orders = cleanup_order_volumes(trader_orders[symbol])
                for order in t_orders:
                    if order.quantity < 0:
                        if halfway:
                            bids = symbol_order_depth.buy_orders.keys()
                            asks = symbol_order_depth.sell_orders.keys()
                            max_bid = max(bids)
                            min_ask = min(asks)
                            if order.price <= statistics.median([max_bid, min_ask]):
                                trades.append(Trade(symbol, order.price, order.quantity, "BOT", "YOU", time))
                            else:
                                print(f'No matches for order {order} at time {time}')
                                print(f'Order depth is {order_depth[order.symbol].__dict__}')
                        else:
                            potential_matches = list(filter(lambda o: o[0] == order.price, symbol_order_depth.buy_orders.items()))
                            if len(potential_matches) > 0:
                                match = potential_matches[0]
                                final_volume = 0
                                if abs(match[1]) > abs(order.quantity):
                                    final_volume = order.quantity
                                else:
                                    #this should be negative
                                    final_volume = -match[1]
                                trades.append(Trade(symbol, order.price, final_volume, "BOT", "YOU", time))
                            else:
                                print(f'No matches for order {order} at time {time}')
                                print(f'Order depth is {order_depth[order.symbol].__dict__}')
                    if order.quantity > 0:
                        if halfway:
                            bids = symbol_order_depth.buy_orders.keys()
                            asks = symbol_order_depth.sell_orders.keys()
                            max_bid = max(bids)
                            min_ask = min(asks)
                            if order.price >= statistics.median([max_bid, min_ask]):
                                trades.append(Trade(symbol, order.price, order.quantity, "YOU", "BOT", time))
                            else:
                                print(f'No matches for order {order} at time {time}')
                                print(f'Order depth is {order_depth[order.symbol].__dict__}')
                        else:
                            potential_matches = list(filter(lambda o: o[0] == order.price, symbol_order_depth.sell_orders.items()))
                            if len(potential_matches) > 0:
                                match = potential_matches[0]
                                final_volume = 0
                                #Match[1] will be negative so needs to be changed to work here
                                if abs(match[1]) > abs(order.quantity):
                                    final_volume = order.quantity
                                else:
                                    final_volume = abs(match[1])
                                trades.append(Trade(symbol, order.price, final_volume, "YOU", "BOT", time))
                            else:
                                print(f'No matches for order {order} at time {time}')
                                print(f'Order depth is {order_depth[order.symbol].__dict__}')
        return trades
                            
csv_header = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
log_header = [
    'Sandbox logs:\n',
    '0 OpenBLAS WARNING - could not determine the L2 cache size on this system, assuming 256k\n',
    'START RequestId: 8ab36ff8-b4e6-42d4-b012-e6ad69c42085 Version: $LATEST\n',
    'END RequestId: 8ab36ff8-b4e6-42d4-b012-e6ad69c42085\n',
    'REPORT RequestId: 8ab36ff8-b4e6-42d4-b012-e6ad69c42085	Duration: 18.73 ms	Billed Duration: 19 ms	Memory Size: 128 MB	Max Memory Used: 94 MB	Init Duration: 1574.09 ms\n',
]

def create_log_file(round: int, day: int, states: dict[int, TradingState], profits_by_symbol: dict[int, dict[str, float]], balance_by_symbol: dict[int, dict[str, float]], trader: Trader):
    file_name = uuid.uuid4()
    timest = datetime.timestamp(datetime.now())
    max_time = max(list(states.keys()))
    log_path = os.path.join('logs', f'{timest}_{file_name}.log')
    with open(log_path, 'w', encoding="utf-8", newline='\n') as f:
        f.writelines(log_header)
        f.write('\n')
        for time, state in states.items():
            if hasattr(trader, 'logger'):
                if hasattr(trader.logger, 'local_logs') != None:
                    if trader.logger.local_logs.get(time) != None:
                        f.write(f'{time} {trader.logger.local_logs[time]}\n')
                        continue
            if time != 0:
                f.write(f'{time}\n')

        f.write(f'\n\n')
        f.write('Submission logs:\n\n\n')
        f.write('Activities log:\n')
        f.write(csv_header)
        for time, state in states.items():
            for symbol in SYMBOLS_BY_ROUND[round]:
                f.write(f'{day};{time};{symbol};')
                bids_length = len(state.order_depths[symbol].buy_orders)
                bids = list(state.order_depths[symbol].buy_orders.items())
                bids_prices = list(state.order_depths[symbol].buy_orders.keys())
                bids_prices.sort()
                asks_length = len(state.order_depths[symbol].sell_orders)
                asks_prices = list(state.order_depths[symbol].sell_orders.keys())
                asks_prices.sort()
                asks = list(state.order_depths[symbol].sell_orders.items())
                if bids_length >= 3:
                    f.write(f'{bids[0][0]};{bids[0][1]};{bids[1][0]};{bids[1][1]};{bids[2][0]};{bids[2][1]};')
                elif bids_length == 2:
                    f.write(f'{bids[0][0]};{bids[0][1]};{bids[1][0]};{bids[1][1]};;;')
                elif bids_length == 1:
                    f.write(f'{bids[0][0]};{bids[0][1]};;;;;')
                else:
                    f.write(f';;;;;;')
                if asks_length >= 3:
                    f.write(f'{asks[0][0]};{asks[0][1]};{asks[1][0]};{asks[1][1]};{asks[2][0]};{asks[2][1]};')
                elif asks_length == 2:
                    f.write(f'{asks[0][0]};{asks[0][1]};{asks[1][0]};{asks[1][1]};;;')
                elif asks_length == 1:
                    f.write(f'{asks[0][0]};{asks[0][1]};;;;;')
                else:
                    f.write(f';;;;;;')
                if len(asks_prices) == 0 or max(bids_prices) == 0:
                    if symbol == 'DOLPHIN_SIGHTINGS':
                        dolphin_sightings = state.observations['DOLPHIN_SIGHTINGS']
                        f.write(f'{dolphin_sightings};{0.0}\n')
                    else:
                        f.write(f'{0};{0.0}\n')
                else:
                    actual_profit = 0.0
                    if symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                            actual_profit = profits_by_symbol[time][symbol] + balance_by_symbol[time][symbol]
                    min_ask = min(asks_prices)
                    max_bid = max(bids_prices)
                    median_price = statistics.median([min_ask, max_bid])
                    f.write(f'{median_price};{actual_profit}\n')
                    if time == max_time:
                        if profits_by_symbol[time].get(symbol) != None:
                            print(f'Final profit for {symbol} = {actual_profit}')
        print(f"\nSimulation on round {round} day {day} for time {max_time} complete")


# Adjust accordingly the round and day to your needs1

if __name__ == "__main__":
    trader = Trader(SPREAD=3,use_macd=False, REGRESSION_SPREAD=1, extra=20, regression_extra=20,
                    linear_regression=[6, [0.0, 0.9487079673973507, 0.04882953537331608, 0.0, 0.001374535182263223, 0.0], 5.475369188194236])
    #simulate_alternative(1, -2, trader, 1*100000, False, True, False)
    
    max_time = int(input("Max timestamp (1-9)->(1-9)(00_000) or exact number): ") or 999000)
    if max_time < 10:
        max_time *= 100000
    round = int(input("Input a round (blank for 4): ") or 4)
    #day = int(input("Input a day (blank for random): ") or random.randint(1, 3))
    names_in = input("With bot names (default: y) (y/n): ")
    names = True
    if 'n' in names_in:
        names = False
    halfway_in = input("Matching orders halfway (default: n) (y/n): ")
    halfway = True
    if 'y' in halfway_in:
        halfway = True
    #print(f"Running simulation on round {round} day {day} for time {max_time}")
    print("Remember to change the trader import")
    
    amethysts, starfruit = 0, 0
    for day in range(-2, 1):
        profits, mids = simulate_alternative(round, day, trader, max_time, names, halfway, False)
        final_key = max(list(profits.keys()))
        amethysts += profits[final_key]['AMETHYSTS']
        starfruit += profits[final_key]['STARFRUIT']
        
        
    print("Simulation complete")
    print(f'AMETHYSTS: {amethysts}, STARFRUIT: {starfruit}')
    #print(profits)
    mids = pd.DataFrame(mids)
    for symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
        symbol_df = mids[symbol]
        symbol_df.to_csv(f'{symbol}_round_{round}_day_{day}.csv', sep=',')
        
        
        
    # Optimising Spread Stationary
    # spread_results = []
    # for extra in range(5, 45, 5):
    #     for spread in [1.5, 2, 2.5, 3]:
    #         new_trader = Trader(use_macd=False, SPREAD=spread, extra=extra)
    #         round = 1
    #         max_time = 2*100000
    #         names = False
    #         halfway = True
            
    #         amethysts, starfruit = 0, 0
    #         for day in range(-2, 1):
    #             profits, mids = simulate_alternative(round, day, new_trader, max_time, names, halfway, False)
    #             # Tally Results
    #             final_key = max(list(profits.keys()))
    #             amethysts += profits[final_key]['AMETHYSTS']
    #             starfruit += profits[final_key]['STARFRUIT']
            
    #         spread_results.append((spread, extra, amethysts, starfruit))
        
    # spread_results.sort(key=lambda x: x[2], reverse=True)
    
    
    
    # #Grid Search MACD Parametrising Test
    # # macd_results = []
    # # macd_limits = [1, 2, 4, 8, 12, 16, 10000]
    # # for macd_limit in macd_limits:
    # #     for i in range(1, 2):
    # #         for j in range(1, 16, 3):
    # #             for x in range(1, 10, 2):
    # #                 new_trader = Trader(use_macd=True, macd_window=[i, j, x], MACD_MAX_ORDER=macd_limit)  
    # #                 round = 1
    # #                 max_time = 2*100000
    # #                 names = False
    # #                 halfway = True
    # #                 amethysts, starfruit = 0, 0
    # #                 for day in range(-2, 1):
    # #                     profits, mids = simulate_alternative(round, day, new_trader, max_time, names, halfway, False)
    # #                     # Tally Results
    # #                     final_key = max(list(profits.keys()))
    # #                     amethysts += profits[final_key]['AMETHYSTS']
    # #                     starfruit += profits[final_key]['STARFRUIT']
                        
    # #                 macd_results.append((i, j, x, amethysts, starfruit))
                
                
    # # # # Sort Results
    # # macd_results.sort(key=lambda x: x[3] + x[4], reverse=True)
   
    
    #Optimising Linear Regression and its Spread
    # linear_results = []
    # linear_models = [[6, [0.0, 0.9487079673973507, 0.04882953537331608, 0.0, 0.001374535182263223, 0.0], 5.475369188194236],
    #                  [6, [0.3318083033107856, 0.3323260816901865, 0.20071972419552198, 0.08160998264750106, 0.04375847244687196, 0.008596169468114855], 5.947416965931552],
    #                  [7, [0.33181448274499276, 0.3322579441835937, 0.2006959852503323, 0.0816130190392255, 0.04381063553086938, 0.008597088354513456, 3.0322461129869954e-05], 5.9437049141924945], 
    #                  [8, [0.33179922480627727, 0.3322624284422393, 0.20068773909075435, 0.08161743456399213, 0.043815663327359734, 0.008606645120145811, 3.01113087998304e-05], 5.944875585296359],
    #                  [9, [0.33129390774635564, 0.33188122935621195, 0.20033869566962287, 0.0815906683815888, 0.043728795471651594, 0.008494921734807537, 1.3031026409534749e-05, 0.0014827454011304133], 5.920918445363895],
    #                  [4, [0.3617986190168745, 0.3448070650356635, 0.20659265289967932, 0.08551401097274297], 6.482179393858132],
    #                  [10, [0.32964853184115245, 0.3313521152127662, 0.2001481102970499, 0.08140985308334522, 0.04368151099364375, 0.008419052421877885, 2.952833553465375e-06, 0.0011741574947806883, 0.002994381339110525], 5.887346632957815]]
    
    # for linear_model in linear_models[:1]:
    #     for spread in [1, 2, 3, 4, 5]:
    #         for reg_extra in range(5,10,5):
    #             new_trader = Trader(use_macd=False, linear_regression=linear_model, regression_extra=reg_extra, REGRESSION_SPREAD=spread)
    #             round = 1
    #             max_time = 2*100000
    #             names = False
    #             halfway = True
    #             amethysts, starfruit = 0, 0
    #             for day in range(-2, 1):
    #                 profits, mids = simulate_alternative(round, day, new_trader, max_time, names, halfway, False)
    #                 # Tally Results
    #                 final_key = max(list(profits.keys()))
    #                 amethysts += profits[final_key]['AMETHYSTS']
    #                 starfruit += profits[final_key]['STARFRUIT']
    #             linear_results.append((spread, reg_extra, amethysts, starfruit))
            
    # linear_results.sort(key=lambda x: x[3], reverse=True)
    
    # # # # # # # # # print(f"MACD Results Parameter Optimization")
    # # # # # # # # # print(macd_results[:min(20, len(macd_results))])
    
    # print(f"Linear Regression Results Parameter Optimization")
    # print(linear_results[:min(20, len(linear_results))])
    
    print()
    
    # print(f"Spread Results and Extra Parameter Optimization")
    # print(spread_results[:10])

    
    
    
