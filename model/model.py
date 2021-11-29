# model.py
# ABM simulation model


# dependencies
import numpy as np
from LOB import LOB, Order, RunOutOfOrderError
from agents import ZI, Fundamentalist, Chartist, Spoofer


# simulation model class
class SimulationModel:
    def __init__(self, num_ZI, ZI_params, num_F, F_params, num_C, C_params, price_series_seed=None, book_seed=None, F_seed=None, spoofing=False, S_params=None):
        # input seed check
        if price_series_seed is None:
            price_series_seed = [299, 301, 299]
        elif not isinstance(price_series_seed, list):
            raise SeedError('price_series_seed', 'list')
        if book_seed is None:
            book_seed = LOB()
            book_seed.add('bid', Order(price=298, size=1))
            book_seed.add('ask', Order(price=300, size=1))
        elif not isinstance(book_seed, LOB):
            raise SeedError('book_seed', 'LOB')
        if F_seed is None:
            F_seed = 300.0
        elif not isinstance(F_seed, float):
            raise SeedError('F_seed', 'float')
        # parameters
        self.t = 0
        self.F = F_seed
        self.price_series = price_series_seed
        self.book = book_seed
        self.pt = self.book.get_mid_price()
        self.spoofing = spoofing
        # agent parameters
        self.num_ZI = num_ZI
        self.ZI_params = ZI_params
        self.num_F = num_F
        self.F_params = F_params
        self.num_C = num_C
        self.C_params = C_params
        #initialize agents
        self.agent_groups = []
        self.agent_groups.append(ZI(model=self, num_agents=num_ZI, ZI_params=ZI_params))
        self.agent_groups.append(Fundamentalist(model=self, num_agents=num_F, F_params=F_params))
        self.agent_groups.append(Chartist(model=self, num_agents=num_C, C_params=C_params))
        if spoofing:
            self.agent_groups.append(Spoofer(model=self, S_params=S_params))
        # observers
        self.num_matching_series = []
        self.spread_series = []
        self.bid_volume = []
        self.ask_volume = []
        self.heuristic_avg = [[], [], []]
        # spoofing debug
        if spoofing:
            self.spoofing_activate_series = []
            self.num_spoofing_order_series = []
            self.spoofing_orders_series = []
            self.spoofing_activated_before_series = []
    
    def step(self):
        # update simulation time
        self.t += 1
        # all agent group submit & cancel orders
        for i, group in enumerate(self.agent_groups):
            group.step()
            if i <=2:
                self.heuristic_avg[i].append(group.heuristic_val_avg)
            
        orderbook_vol_temp = self.book.get_volume('bid') + self.book.get_volume('ask')
        # record orderbook volume before
        # order book matching
        self.book.matching()
        cur_num_matching = orderbook_vol_temp - (self.book.get_volume('bid') + self.book.get_volume('ask'))
        # update price series & obs
        try:
            self.pt = self.book.get_mid_price()
        except RunOutOfOrderError:
            pass
        self.price_series.append(self.pt)
        self.num_matching_series.append(cur_num_matching)
        try:
            self.spread_series.append(np.abs(self.book.get_best_ask_price() - self.book.get_best_bid_price()) )
        except RunOutOfOrderError:
            self.spread_series.append(np.nan)
        try:
            self.bid_volume.append(self.book.get_volume('bid'))
        except RunOutOfOrderError:
            self.bid_volume.append(np.nan)
        try:
            self.ask_volume.append(self.book.get_volume('ask'))
        except RunOutOfOrderError:
            self.ask_volume.append(np.nan)
        # record spoofing order series
        if self.spoofing:
            self.num_spoofing_order_series.append(len(self.agent_groups[-1].spoofing_orders))
            self.spoofing_orders_series.append(self.agent_groups[-1].spoofing_orders)
            self.spoofing_activate_series.append(self.agent_groups[-1].spoofing_activated_now)
            self.spoofing_activated_before_series.append(self.agent_groups[-1].spoofing_activated_before)
    
    def get_return_series(self):
        price_series = np.array(self.price_series)
        return np.diff(price_series) / price_series[:-1]


# errors
class SeedError(Exception):
    def __init__(self, error_seed, correct_seed_type):
        self.error_seed = error_seed
        self.correct_seed_type = correct_seed_type
    
    def __str__(self):
        return f'{self.error_seed} does not match with expected seed type: {self.correct_seed_type}.'
