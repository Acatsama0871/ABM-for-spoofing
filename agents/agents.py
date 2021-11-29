# agents.py
# The agents class from ABM model


# dependencies
import numpy as np
from abc import ABC, abstractmethod
from LOB.LOB import Order, OrderNotFoundError


# helper functions
# window evaluation function
def window_mean(series, L, ignore_insufficient_lengths=True):
    if len(series) < L and not ignore_insufficient_lengths:
        raise WindowLengthError

    return np.diff(series[-L:]).mean()

# price calculation helper function
def price_helperfunc(p_lambda, lognormal, sign):
    lognormal = lognormal / p_lambda
    if sign == 1:
        return (2.0 - (lognormal))
    elif sign == -1:
        return lognormal
    elif sign == 0:
        return np.nan
    else:
        raise OrderPriceSignError(sign=sign)
price_helperfunc = np.vectorize(price_helperfunc)

# order price generator
class OrderPriceGenerator:
    def __init__(self, sigs):
        self.sigs = sigs
        self.rng = np.random.default_rng()
    
    def __call__(self, cur_price, signs, lambdas):
        lognormals = self.rng.lognormal(sigma=self.sigs)

        return np.round(cur_price * price_helperfunc(p_lambda=lambdas, lognormal=lognormals, sign=signs), 2)

# order size generator
class OrderSizeGenerator:
    def __init__(self, sigma=2.5, clipping_lower=1, clipping_upper=10):
        self.sigma = sigma
        self.rng = np.random.default_rng()
        self.clipping_lower = clipping_lower
        self.clipping_upper = clipping_upper
    def sample(self, size, gamma=1, resample_ratio=0.5):
        # lognormal randon number
        lognormal_random = self.rng.lognormal(mean=0.0, sigma=self.sigma, size=size)
        order_size = np.round(lognormal_random)
        order_size = order_size[(order_size < (self.clipping_upper + 1)) & (order_size != (self.clipping_lower - 1))]
        # get remaining sample size
        remaining_sample_size = size - len(order_size)
        # if there is remaining sample size
        if remaining_sample_size != 0:
            while True:
                resample_lognormal = self.rng.lognormal(mean=0.0, sigma=self.sigma, size=int(np.round(resample_ratio * size)))
                resample_order_size = np.round(resample_lognormal)
                resample_order_size = resample_order_size[(resample_order_size < (self.clipping_upper + 1)) & (resample_order_size != (self.clipping_lower - 1))]
                # if resampled is larger than we needed: randon sample to the size we need
                if len(resample_order_size) > remaining_sample_size:
                    resample_order_size = self.rng.choice(resample_order_size, size=remaining_sample_size, replace=False)
                # recalculate the remaining size & add sampled to order size
                remaining_sample_size -= len(resample_order_size)
                order_size = np.concatenate((order_size, resample_order_size), axis=0)
                # if remaining sample size -> 0, break
                if remaining_sample_size == 0:
                    break
        # add gamma
        order_size = order_size + gamma
        
        return order_size

# random heuristic generator
class RandomHeuristic:
    def __init__(self, num_agents, sigs):
        self.num_agents = num_agents
        self.sigs = sigs
        self.rng = np.random.default_rng()

    def __call__(self):
        return self.rng.normal(loc=0, scale=self.sigs)


# agent classes
# base agent class
class Agent(ABC):
    def __init__(self, ids, model):
        self.ids = ids
        self.model = model

    @abstractmethod
    def update_market_info(self):
        pass

    @abstractmethod
    def update_order_status(self):
        pass

    @abstractmethod
    def submit_orders(self):
        pass

    @abstractmethod
    def cancel_orders(self):
        pass

# heuristic agent base class
class HeuristicAgentBase(Agent):
    def __init__(self, ids, model):
        super(HeuristicAgentBase, self).__init__(ids, model)
        # model obs
        self.t = model.t  # current time
        self.F = model.F  # current fundamental value
        self.pt = model.pt  # current mid price

    @abstractmethod
    def heuristic(self):
        self.update_market_info()

    def update_market_info(self):
        self.t = self.model.t
        self.F = self.model.F
        self.pt = self.model.pt
    
    @abstractmethod
    def update_order_status(self):
        pass
    
    @abstractmethod
    def submit_orders(self):
        pass
    
    def cancel_orders(self):
        # update status
        to_be_canceled_bid, to_be_canceled_ask = self.update_order_status()
        # cancel orders beyond life span
        self.model.book.remove('bid', to_be_canceled_bid)
        self.model.book.remove('ask', to_be_canceled_ask)
    
    def step(self):
        self.submit_orders()
        self.cancel_orders()

class RandomArrivalAgent(HeuristicAgentBase):
    def __init__(self, model, num_agents, agent_params):
        self.order_params = agent_params
        # rng
        self.rng = np.random.default_rng()
        # agents and their ids
        self.num_agents = num_agents
        ids = list(range(num_agents))
        # order arrival & order life parameters
        self.average_order_arrival_durations = agent_params['order_duration']
        self.average_orders_life = agent_params['order_life']
        self.next_order_arrival_wait_time = np.zeros(self.num_agents)
        self.next_order_arrival_time = self.rng.exponential(scale=self.average_order_arrival_durations, size=num_agents) 
        # order size parameters & generator
        self.order_size_sigma = agent_params['order_size_sigma']
        self.order_size_lower = agent_params['order_size_lower']
        self.order_size_upper = agent_params['order_size_upper']
        self.order_size_gamma = agent_params['order_size_gamma']
        self.size_generator = OrderSizeGenerator(sigma=self.order_size_sigma, clipping_lower=self.order_size_lower, clipping_upper=self.order_size_upper)
        # order price parameters & generator
        self.order_price_sigma = np.full(self.num_agents, agent_params['order_price_sigma'])
        self.lambdas = np.full(self.num_agents, agent_params['order_price_lambda'])  # can change to different if needed
        self.price_generator = OrderPriceGenerator(sigs=self.order_price_sigma)
        # initialize base class
        super(RandomArrivalAgent, self).__init__(ids, model)
        # submitted orders
        self.submitted_orders = {cur_id: [] for cur_id in self.ids}
        # observer
        self.num_bid_miss = 0  # number of times where bid price < 0(then the agent will not submit the order)
        self.bid_prices = []  # deprecated for now
        self.ask_prices = []  # deprecated for now
        self.agents_action_record = []  # 1 for bid and -1 for ask, 0 for no action
        self.agents_order_price_record = []  # submittted order record
        self.agents_traded_action_record = []  # 1 for bid and -1 for ask, 0 for no action
        self.agents_traded_price_record = []  # traded price record
        self.agents_traded_order_size_record = []

    
    @abstractmethod
    def heuristic(self):
        self.update_market_info()
    
    def update_order_status(self):
        consumes = []  # orders will be removed from local record
        to_be_canceled_bid = []  # order reach its life span will be removed from bid side
        to_be_canceled_ask = []  # order reach its life span will be removed from ask side
        
        # update cur matched
        for cur_id in self.submitted_orders:  # get agents' id
            for cur_order in self.submitted_orders[cur_id]:
                # if fully filled
                if not self.model.book.inquire(cur_order['direction'], cur_order['ID']):
                    cur_order['cur_matched'] = cur_order['pre_order_size']
                # if not match or partially filled
                else:
                    if cur_order['direction'] == 'bid':
                        cur_order['cur_matched'] = cur_order['pre_order_size'] - self.model.book.bid_side.inquire_remaining_size(cur_order['ID'])
                        cur_order['pre_order_size'] = self.model.book.bid_side.inquire_remaining_size(cur_order['ID'])
                    elif cur_order['direction'] == 'ask':
                        cur_order['cur_matched'] = cur_order['pre_order_size'] - self.model.book.ask_side.inquire_remaining_size(cur_order['ID'])
                        cur_order['pre_order_size'] = self.model.book.ask_side.inquire_remaining_size(cur_order['ID'])
            
        # get each agent's submitted orders
        cur_actions = [0] * self.num_agents
        cur_order_prices = [0] * self.num_agents
        cur_order_size = [0] * self.num_agents
        for cur_id in self.submitted_orders:
            # check each order's status
            for cur_order in self.submitted_orders[cur_id]:
                # check if traded
                if not self.model.book.inquire(cur_order['direction'], cur_order['ID']):
                    consumes.append(cur_order['ID'])  # if traded add to consume
                    if cur_order['direction'] == 'bid':
                        cur_actions[cur_id] = 1
                        cur_order_prices[cur_id] = cur_order['market_order_price']
                        cur_order_size[cur_id] = cur_order['cur_matched']
                    elif cur_order['direction'] == 'ask':
                        if cur_order['market_order_price'] != 'None':
                            cur_actions[cur_id] = -1
                            cur_order_prices[cur_id] = cur_order['market_order_price']
                        cur_order_size[cur_id] = cur_order['cur_matched']
                # check if order reaches its life span
                elif cur_order['t'] > cur_order['life']:
                    if cur_order['direction'] == 'bid':
                        cur_matched = cur_order['cur_matched']
                        if cur_matched != 0:
                            cur_actions[cur_id] = 1
                            cur_order_prices[cur_id] = cur_order['market_order_price']
                            cur_order_size[cur_id] = cur_order['cur_matched']
                        to_be_canceled_bid.append(cur_order['ID'])
                    else:
                        cur_matched = cur_order['cur_matched']
                        if cur_matched != 0:
                            cur_actions[cur_id] = -1
                            cur_order_prices[cur_id] = cur_order['market_order_price']
                            cur_order_size[cur_id] = cur_order['cur_matched']
                        to_be_canceled_ask.append(cur_order['ID'])
                    consumes.append(cur_order['ID'])
                # if not expired or traded, update t += 1
                else:
                    cur_order['t'] += 1
                    if cur_order['direction'] == 'bid':
                        cur_matched = cur_order['cur_matched']
                        if cur_matched != 0:
                            cur_actions[cur_id] = 1
                            cur_order_prices[cur_id] = cur_order['market_order_price']
                            cur_order_size[cur_id] = cur_order['cur_matched']
                    else:
                        cur_matched = cur_order['cur_matched']
                        if cur_matched != 0:
                            cur_actions[cur_id] = -1
                            cur_order_prices[cur_id] = cur_order['market_order_price']
                            cur_order_size[cur_id] = cur_order['cur_matched']
            # remove consumes from submitted orders
            self.submitted_orders[cur_id] = [cur_order for cur_order in self.submitted_orders[cur_id] if cur_order['ID'] not in consumes]
        
        self.agents_traded_action_record.append(cur_actions)
        self.agents_traded_price_record.append(cur_order_prices)
        self.agents_traded_order_size_record.append(cur_order_size)
        
        return to_be_canceled_bid, to_be_canceled_ask 
    
    def submit_orders(self):
        # get current trading sign
        cur_signs = self.heuristic()
        # get current trading price
        cur_prices = self.price_generator(cur_price=self.pt, signs=cur_signs, lambdas=self.lambdas)
        # get current order size
        cur_order_sizes = self.size_generator.sample(size=self.num_agents, gamma=self.order_size_gamma)
        # get current life
        cur_orders_life = self.rng.exponential(scale=self.average_orders_life, size=self.num_agents)
        # get current next arrival time
        cur_orders_arrival_time = self.rng.exponential(scale=self.average_order_arrival_durations, size=self.num_agents)  # may not be used
        # construct and submit orders
        submit_list_bid = []
        submit_list_ask = []
        cur_agents_actions = [0] * self.num_agents
        cur_agents_order_price_record = [0] * self.num_agents
        for cur_id in self.submitted_orders:
            # if wait enough time to submit order
            if self.next_order_arrival_wait_time[cur_id] > self.next_order_arrival_time[cur_id]:
                # submit orders
                cur_order_price = cur_prices[cur_id]
                cur_order_life = cur_orders_life[cur_id]
                cur_order_size = cur_order_sizes[cur_id]
                cur_order = Order(price=cur_order_price, size=cur_order_size)
                if cur_signs[cur_id] == 1:  # if bid
                    if cur_order_price >= 0:
                        submit_list_bid.append(cur_order)  # wait for submitting to order book
                        if cur_order_price > self.model.book.get_best_ask_price():
                            self.submitted_orders[cur_id].append({'ID': cur_order.ID, 'direction': 'bid', 'price': cur_order.price, 'quantity': cur_order.size, 't': 1, 'life': cur_order_life, 'market_order_price': self.pt, 'pre_order_size': cur_order.size, 'cur_matched': 0})  # record to submitted orders
                        else:
                            self.submitted_orders[cur_id].append({'ID': cur_order.ID, 'direction': 'bid', 'price': cur_order.price, 'quantity': cur_order.size, 't': 1, 'life': cur_order_life, 'market_order_price': cur_order.price, 'pre_order_size': cur_order.size, 'cur_matched': 0})  # record to submitted orders
                        # profit record
                        cur_agents_actions[cur_id] = 1
                        # print(cur_agents_actions[cur_id])
                        cur_agents_order_price_record[cur_id] = cur_order_price
                    else:
                        self.num_bid_miss += 1
                else:  # if ask
                    # print('2')
                    if cur_order_price <= self.pt * 2:
                        submit_list_ask.append(cur_order)
                        if cur_order_price < self.model.book.get_best_bid_price():
                            self.submitted_orders[cur_id].append({'ID': cur_order.ID, 'direction': 'ask', 'price': cur_order.price, 'quantity': cur_order.size, 't': 1, 'life': cur_order_life, 'market_order_price': self.pt, 'pre_order_size': cur_order.size, 'cur_matched': 0})
                        else:
                            self.submitted_orders[cur_id].append({'ID': cur_order.ID, 'direction': 'ask', 'price': cur_order.price, 'quantity': cur_order.size, 't': 1, 'life': cur_order_life, 'market_order_price': cur_order_price, 'pre_order_size': cur_order.size, 'cur_matched': 0})
                        # profit record
                        cur_agents_actions[cur_id] = -1
                        cur_agents_order_price_record[cur_id] = cur_order_price
                # reset arrival waiting time & resample next arrival duration
                self.next_order_arrival_wait_time[cur_id] = 0
                self.next_order_arrival_time[cur_id] = cur_orders_arrival_time[cur_id]
            else:
                self.next_order_arrival_wait_time[cur_id] += 1
        # submit orders to book
        self.model.book.add('bid', submit_list_bid)
        self.model.book.add('ask', submit_list_ask)
        cur_bid_prices = [order.price for order in submit_list_bid]
        cur_ask_prices = [order.price for order in submit_list_ask]
        # profit record
        self.bid_prices.append(cur_bid_prices)
        self.ask_prices.append(cur_ask_prices)
        self.agents_action_record.append(cur_agents_actions)
        self.agents_order_price_record.append(cur_agents_order_price_record)

# ZI trader
class ZI(HeuristicAgentBase):
    def __init__(self, model, num_agents, ZI_params):
        self.num_agents = num_agents
        self.order_size_multiplier = ZI_params['order_size_multiplier']
        # generate agents' ids & order life series
        ids = list(range(num_agents))
        self.orders_life = np.ones(num_agents) * 2
        # initialize base class
        super(ZI, self).__init__(ids=ids, model=model)
        # order price params & price generator functions & order size(fixed for ZI agent)
        self.rng = np.random.default_rng()
        self.lambdas = np.full(self.num_agents, ZI_params['order_price_lambda'])
        self.sigs = np.full(self.num_agents, ZI_params['order_price_sig'])
        self.price_generator = OrderPriceGenerator(sigs=self.sigs)
        # submitted orders
        self.submitted_orders = {cur_id: [] for cur_id in self.ids}
        # observer
        self.num_bid_miss = 0
        self.heuristic_val_avg = 0

    def heuristic(self):
        self.update_market_info()
        return self.rng.choice(a=[-1, 1], size=self.num_agents, replace=True, p=[0.5, 0.5])  #buy and sell with equal probability
    
    def update_order_status(self):
        consumes = []  # orders will be removed from local record
        to_be_canceled_bid = []  # order reach its life span will be removed from bid side
        to_be_canceled_ask = []  # order reach its life span will be removed from ask side

        # get each agent's submitted orders
        for cur_id in self.submitted_orders:
            # check each order's status
            for cur_order in self.submitted_orders[cur_id]:
                # check if traded
                if not self.model.book.inquire(cur_order['direction'], cur_order['ID']):
                    consumes.append(cur_order['ID'])  # if traded add to consume
                # check if order reaches its life span
                elif cur_order['t'] > self.orders_life[cur_id]:
                    if cur_order['direction'] == 'bid':
                        to_be_canceled_bid.append(cur_order['ID'])
                    else:
                        to_be_canceled_ask.append(cur_order['ID'])
                    consumes.append(cur_order['ID'])
                # if not expired or traded, update t += 1
                else:
                    cur_order['t'] += 1
            # remove consumes from submitted orders
            self.submitted_orders[cur_id] = [cur_order for cur_order in self.submitted_orders[cur_id] if cur_order['ID'] not in consumes]

        return to_be_canceled_bid, to_be_canceled_ask 

    def submit_orders(self):
        # get current trading sign
        cur_signs = self.heuristic()
        # get current trading price
        cur_prices = self.price_generator(cur_price=self.pt, signs=cur_signs, lambdas=self.lambdas)
        # construct and submit orders
        submit_list_bid = []
        submit_list_ask = []
        for cur_id in self.submitted_orders:
            cur_order_price = cur_prices[cur_id]
            cur_order = Order(price=cur_order_price, size=self.order_size_multiplier)
            if cur_signs[cur_id] == 1:  # if bid
                if cur_order_price > 0:
                    submit_list_bid.append(cur_order)  # wait for submitting to order book
                    self.submitted_orders[cur_id].append({'ID': cur_order.ID, 'direction': 'bid', 'price': cur_order.price, 'quantity': cur_order.size, 't': 1})  # record to submitted orders
                else:
                    self.num_bid_miss += 1
            else:  # if ask
                if cur_order_price <= self.pt * 2:
                    submit_list_ask.append(cur_order)
                    self.submitted_orders[cur_id].append({'ID': cur_order.ID, 'direction': 'ask', 'price': cur_order.price, 'quantity': cur_order.size, 't': 1})
        # submit orders to book
        self.model.book.add('bid', submit_list_bid)
        self.model.book.add('ask', submit_list_ask)
    
    # Fundamentalist
class Fundamentalist(RandomArrivalAgent):
    def __init__(self, model, num_agents, F_params):
        # initialize parent class
        super(Fundamentalist, self).__init__(model, num_agents, F_params)
        # F parameters
        self.a_lower = F_params['a_lower']
        self.a_upper = F_params['a_upper']
        self.a = self.rng.uniform(self.a_lower, self.a_upper, size=self.num_agents)
        self.h_sig_lower = F_params['h_sig_lower']
        self.h_sig_upper = F_params['h_sig_upper']
        self.h_sig = self.rng.uniform(self.h_sig_lower, self.h_sig_upper, size=self.num_agents)
        # observer
        self.heuristic_val = np.zeros(num_agents)
        self.heuristic_val_avg = np.mean(self.heuristic_val)
    
    def heuristic(self):
        # update market info
        self.update_market_info()
        # calculate the heuristic
        self.heuristic_val = self.a * (self.F - self.pt) + self.h_sig * self.rng.normal(size=self.num_agents)
        # return trading actions
        cur_actions = np.zeros(self.num_agents)
        cur_actions[self.heuristic_val > 0] = 1
        cur_actions[self.heuristic_val < 0] = -1
        # update heuristic value average
        self.heuristic_val_avg = np.mean(self.heuristic_val)
        
        return cur_actions
    
# Chartist
class Chartist(RandomArrivalAgent):
    def __init__(self, model, num_agents, C_params):
        # initialize parent class
        super(Chartist, self).__init__(model, num_agents, C_params)
        # C parameters
        self.L_lower = C_params['L_lower']
        self.L_upper = C_params['L_upper']
        self.L = self.rng.integers(self.L_lower, self.L_upper, size=num_agents)
        self.c_lower = C_params['c_lower']
        self.c_upper = C_params['c_upper']
        self.c = self.rng.uniform(self.c_lower, self.c_upper, size=num_agents)
        self.h_sig_lower = C_params['h_sig_lower']
        self.h_sig_upper = C_params['h_sig_upper']
        self.h_sig = self.rng.uniform(self.h_sig_lower, self.h_sig_upper, size=self.num_agents)
        # imbalance sense
        self.d_lower = C_params['d_lower']
        self.d_upper = C_params['d_upper']
        self.d = self.rng.uniform(self.d_lower, self.d_upper, size=self.num_agents)
        self.Tsp_lower = C_params['Tsp_lower']
        self.Tsp_upper = C_params['Tsp_upper']
        self.Tsp = self.rng.uniform(self.Tsp_lower, self.Tsp_upper, size=self.num_agents)
        # add window mean
        self.window_evaluation = np.zeros(num_agents)
        for i in range(num_agents):
            self.window_evaluation[i] = window_mean(series=self.model.price_series, L=self.L[i])
        # add observer
        self.heuristic_val = np.zeros(num_agents)
        self.heuristic_val_avg = np.mean(self.heuristic_val)
    
    def update_market_info(self):
        self.t = self.model.t
        self.F = self.model.F
        self.pt = self.model.pt
        # update window mean
        for i in range(self.num_agents):
            self.window_evaluation[i] = window_mean(series=self.model.price_series, L=self.L[i])
    
    def heuristic(self):
        # update market info
        self.update_market_info()
        # calculate current heuristic
        if self.model.book.get_volume('bid') > self.model.book.get_volume('ask'):
            # cur_imbalance_ratio = np.abs(self.model.book.get_volume('bid') / self.model.book.get_volume('ask') - 1)
            cur_imbalance_ratio = self.model.book.get_volume('bid') / self.model.book.get_volume('ask') - 1
            cur_imblance_heurstic = self.d * np.maximum(0, (cur_imbalance_ratio - self.Tsp))
        elif self.model.book.get_volume('ask') > self.model.book.get_volume('bid'):
            # cur_imbalance_ratio = np.abs(self.model.book.get_volume('ask') / self.model.book.get_volume('bid') - 1)
            cur_imbalance_ratio = self.model.book.get_volume('ask') / self.model.book.get_volume('bid') - 1
            cur_imblance_heurstic = -self.d * np.maximum(0, (cur_imbalance_ratio - self.Tsp))
        else:
            cur_imblance_heurstic = 0.0
        self.heuristic_val = self.c * self.window_evaluation + self.h_sig * self.rng.normal(size=self.num_agents) + cur_imblance_heurstic
        # return trading trading actions
        cur_actions = np.zeros(self.num_agents)
        cur_actions[self.heuristic_val > 0] = 1
        cur_actions[self.heuristic_val < 0] = -1
        # update heuristic value average
        self.heuristic_val_avg = np.mean(self.heuristic_val)
        
        return cur_actions
    

# spoofer parameters:
# manipulate direction: price manipulate direction
# arrival time
# manipulate order price
# order size increment speed
# order size ratio: spoofing order size = current volume * order size ratio
# order insert location: spoofing order insert location relative to current price
# order revising duration: duration of order revising 
class Spoofer(Agent):
    def __init__(self, model, S_params):
        # initialize parent class
        super(Spoofer, self).__init__(0, model)
        # spoofer parameters
        self.direction = S_params['direction']
        self.arrival_time = S_params['arrival_time']
        self.manipulate_price = S_params['manipulate_price']
        self.order_size_increment_step = S_params['order_size_increment_step']
        self.order_size_ratio = S_params['order_size_ratio']
        self.order_insert_location = S_params['order_insert_location']
        self.order_revising_duration = S_params['order_revising_duration']
        # spoofing steps
        if self.order_size_increment_step == 'None' or (self.order_size_increment_step > self.order_size_ratio):
            self.order_size_ratio_series = [self.order_size_ratio]
        else:
            self.order_size_ratio_series = np.arange(self.order_size_increment_step, self.order_size_ratio + self.order_size_increment_step, self.order_size_increment_step).tolist()
        # extra market obs
        self.t = self.model.t
        self.best_bid = self.model.book.get_best_bid_price()
        self.best_ask = self.model.book.get_best_ask_price()
        self.pt = self.model.pt
        self.vol = self.model.book.get_volume('bid') + self.model.book.get_volume('ask')
        # behavior control
        self.spoofing_activated_before = False
        self.spoofing_activated_now = False
        self.revising_wait_time = 0
        # orders
        self.spoofing_orders = []
        self.manipulating_order = None
        self.manipulating_order_direction = None
        self.num_not_found = 0
        # observer
        self.best_bid_obs = []
        self.spoofing_order_trade_time = []
    
    def update_market_info(self):
        # update market info
        self.t = self.model.t
        self.pt = self.model.pt
        self.best_bid = self.model.book.get_best_bid_price()
        self.best_bid_obs.append(self.best_bid)
        self.best_ask = self.model.book.get_best_ask_price()
        self.vol = self.model.book.get_volume('bid') + self.model.book.get_volume('ask')
        # check if it is time to activate spoofing
        if self.t == self.arrival_time:
            self.spoofing_activated_now = True
            
    def update_order_status(self):
        # if there is no manipulating order, submit one to book
        if self.manipulating_order is None:
            # if manipulating direction is up, then manipulating order is ask
            if self.direction == 'up':
                the_order = Order(price=self.manipulate_price, size=1)
                self.manipulating_order_direction = 'ask'
                self.manipulating_order = the_order
                self.model.book.add('ask', the_order)
            # if manipulating direction is down, then manipulating order is bid
            elif self.direction == 'down':
                the_order = Order(price=self.manipulate_price, size=1)
                self.manipulating_order_direction = 'bid'
                self.manipulating_order = the_order
                self.model.book.add('bid', the_order)
            else:
                raise ValueError('invalid direction')
        # if manipulating order is submitted, then check if it is traded
        else:
            # if traded, stop spoofing
            if not self.model.book.inquire(direction=self.manipulating_order_direction, ID=self.manipulating_order.ID):
                # alter attributes
                self.spoofing_activated_before = True
                self.spoofing_activated_now = False
                # cancel spoofing orders
                self.cancel_orders()
                self.spoofing_order_trade_time.append(self.t)
            # if not traded, cancel oder spoofing orders and submit new orders
            else:
                if self.revising_wait_time >= self.order_revising_duration:
                    self.cancel_orders()
                    self.submit_orders()
                    self.revising_wait_time = 0
                else:
                    self.revising_wait_time += 1
    
    # submit spoofing orders
    def submit_orders(self):
        try:
            cur_ratio = self.order_size_ratio_series.pop(0)
        except IndexError:
            cur_ratio = self.order_size_ratio
        cur_order_size = int(np.round(cur_ratio * self.vol))
        cur_insert_location = np.round(self.pt * self.order_insert_location, 2)
        if self.direction == 'up':
            cur_insert_location = np.round(self.best_bid * self.order_insert_location, 2)
        elif self.direction == 'down':
            cur_insert_location = np.round(self.best_ask * self.order_insert_location, 2)
        # if manipulating direction is up, then spoofing order is bid
        if self.direction == 'up':
            for _ in range(cur_order_size):
                the_order = Order(price=cur_insert_location, size=1)
                self.spoofing_orders.append(the_order)
                self.model.book.add('bid', the_order)
        # if manipulating direction is down, then spoofing order is ask
        elif self.direction == 'down':
            for _ in range(cur_order_size):
                the_order = Order(price=cur_insert_location, size=1)
                self.spoofing_orders.append(the_order)
                self.model.book.add('ask', the_order)
        else:
            raise ValueError('invalid direction')
    
    # cancel spoofing orders
    def cancel_orders(self):
        for cur_order in self.spoofing_orders:
            if self.direction == 'up':
                try:
                    self.model.book.remove('bid', cur_order.ID)
                except OrderNotFoundError:
                    self.num_not_found += 1
            elif self.direction == 'down':
                try:
                    self.model.book.remove('ask', cur_order.ID)
                except OrderNotFoundError:
                    self.num_not_found += 1
            else:
                raise ValueError('invalid direction')
        self.spoofing_orders = []
    
    def step(self):
        # if not spoofing before
        if not self.spoofing_activated_before:
            self.update_market_info()
            if self.spoofing_activated_now:
                self.update_order_status()

# errors
class WindowLengthError(Exception):
    def __str__(self):
        return "Evaluation length is less than the price series length."

class OrderPriceSignError(Exception):
    def __init__(self, sign):
        self.sign = sign

    def __str__(self):
        return f"Sign {self.sign} does not belong to +1(bid), -1(ask) or 0(hold)."
