import os
import numpy as np
from tqdm import tqdm
from model import SimulationModel

num_obs = 100
f_sharpe_save = []
c_sharpe_save = []
s_sharpe_save = []


def get_spoofing_interval(price_series, spoofing_start=1000):
    # reach_spoofing_index = np.argmax(prices > 500)
    reach_spoofing_index = np.argmax(price_series > 500)
    return_to_normal_index = np.argmax((np.abs(price_series[reach_spoofing_index:] - 300)) < 10)
    return reach_spoofing_index + return_to_normal_index

def sharpe_one_agent(agent_signal, agent_shares, agent_prices, agent_action_time, prices, evaluate_step=10, ret=False):
    agent_signal = agent_signal.copy()
    agent_shares = agent_shares.copy()
    agent_prices = agent_prices.copy()
    
    # if there is no error
    if len(agent_signal) == 0:
        if ret:
            return np.nan, []
        else:
            return np.nan
    agent_action_time = agent_action_time.copy()
    prices = prices.copy()
    agent_holding = 0
    agent_wealth = 0
    ret_series = []
    timeline = list(range(0, len(prices)))
    
    for cur_t in timeline:
        # print(f'Current time: {cur_t}')
        # print(agent_holding)
        if cur_t in agent_action_time:
            cur_signal = agent_signal.pop(0)
            cur_shares = agent_shares.pop(0)
            cur_price = agent_prices.pop(0)
            agent_holding += cur_shares * cur_signal
            agent_wealth += cur_shares * cur_price * cur_signal
        if cur_t % evaluate_step == 0:
            if agent_wealth == 0:
                ret_series.append(np.nan)
            else:
                if agent_holding > 0:
                    cur_agent_wealth = agent_holding * prices[cur_t]
                    ret_series.append(cur_agent_wealth/agent_wealth - 1)
                else:
                    cur_agent_wealth = agent_holding * prices[cur_t]
                    ret_series.append(1 - cur_agent_wealth/agent_wealth)
    
    # calculate sharpe
    if np.nanstd(ret_series) == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = np.nanmean(ret_series) / np.nanstd(ret_series)
    
    if ret:
        return sharpe_ratio, ret_series
    else:
        return sharpe_ratio

def sharpe_agent_group(agent_signals, agent_shares, agent_prices, prices, sharpe_series=False, evaluate_step=1):
    num_agent = agent_signals.shape[1]
    sharpe_ratios = []
    
    for cur_id in range(num_agent):
        # print(f'ID: {cur_id},')
        cur_signal = agent_signals[:, cur_id]
        cur_action_time = list(np.where(cur_signal != 0)[0])
        cur_signal = list(cur_signal[cur_signal != 0])
        cur_shares = agent_shares[:, cur_id]
        cur_shares = list(cur_shares[cur_shares != 0])
        cur_order_prices= agent_prices[:, cur_id]
        cur_order_prices = list(cur_order_prices[cur_order_prices != 0])
        
        cur_sharpe=sharpe_one_agent(agent_signal=cur_signal,
                                    agent_shares=cur_shares,
                                    agent_prices=cur_order_prices,
                                    agent_action_time=cur_action_time,
                                    prices=prices,
                                    evaluate_step=evaluate_step)
        sharpe_ratios.append(cur_sharpe)
    
    if sharpe_series:
        return np.nanmean(sharpe_ratios), sharpe_ratios
    else:
        return np.nanmean(sharpe_ratios)

for _ in tqdm(range(num_obs)):
    try: 
        # run model
        # number of agents
        num_ZI = 100  # number of ZI agents
        num_F = 200  # number of fundamentalists agents
        num_C = 200  # number of chartists agents

        # agent parameters
        ZI_params = {'order_price_lambda': 0.8, 'order_price_sig': 0.25, 'order_size_multiplier': 1}
        F_params = {'a_lower': 0.1, 'a_upper': 0.15, 'h_sig_lower': 1.0, 'h_sig_upper': 2.0, 'order_duration': 15, 'order_life': 5, 'order_size_sigma': 2.5, 'order_size_lower': 1.0, 'order_size_upper': 10.0, 'order_price_sigma': 0.25, 'order_price_lambda': 0.8, 'order_size_gamma': 3.0}
        C_params = {'c_lower': 0.0025, 'c_upper': 0.01, 'L_lower': 3, 'L_upper': 10, 'h_sig_lower':0.1, 'h_sig_upper': 0.2, 'order_duration': 5, 'order_life': 5, 'order_size_sigma': 2.5, 'order_size_lower': 1.0, 'order_size_upper': 10.0, 'order_price_sigma': 0.25, 'order_price_lambda': 0.8, 'order_size_gamma': 3.0, 'd_lower': 0.08, 'd_upper': 0.1, 'Tsp_lower': 1.4, 'Tsp_upper': 2.0}
        S_params = {'direction': 'up', 'arrival_time': 1000, 'manipulate_price': 500, 'order_size_ratio': 0.6, 'order_insert_location': 0.7, 'order_revising_duration': 0, 'order_size_increment_step': 0.01}

        # number of simulation times
        num_sim_times = 2000

        # run simulation
        model = SimulationModel(num_ZI=num_ZI, 
                                num_F=num_F,
                                num_C=num_C,
                                ZI_params=ZI_params,
                                F_params=F_params,
                                C_params=C_params,
                                spoofing=True,
                                S_params=S_params)

        for u in range(num_sim_times):
            model.step()

        price_series = np.array(model.price_series)
        f_actions = np.array(model.agent_groups[1].agents_traded_action_record)
        f_order_prices = np.array(model.agent_groups[1].agents_traded_price_record)
        f_order_sizes = np.array(model.agent_groups[1].agents_traded_order_size_record)
        c_actions = np.array(model.agent_groups[2].agents_traded_action_record)
        c_order_prices = np.array(model.agent_groups[2].agents_traded_price_record)
        c_order_sizes = np.array(model.agent_groups[2].agents_traded_order_size_record)
        return_series = np.array(model.get_return_series())
        spread_series = np.array(model.spread_series)
        ask_vol_series = np.array(model.ask_volume)
        bid_vol_series = np.array(model.bid_volume)
        f_h = np.array(model.heuristic_avg[1])
        c_h = np.array(model.heuristic_avg[2])
        spoofing_order_trade_time = np.array(model.agent_groups[3].spoofing_order_trade_time)
        
        # sharpe prepare
        prices = price_series[3:]  # num of seed is 3
        spoofing_end = get_spoofing_interval(price_series)
        prices = price_series[1000:spoofing_end]
        # fundamentalist
        f_order_action = f_actions[1000:spoofing_end, :]
        f_order_shares = f_order_sizes[1000:spoofing_end, :]
        f_order_prices = f_order_prices[1000:spoofing_end, :]
        # chartist
        c_order_action = c_actions[1000:spoofing_end, :]
        c_order_shares = c_order_sizes[1000:spoofing_end, :]
        c_order_prices = c_order_prices[1000:spoofing_end, :]
        # create spoofing agent action, shares, prices
        s_signal = [1, -1]
        s_shares = [1.0, 1.0]
        s_prices = [prices[0], 500]
        s_action_time = [0, spoofing_order_trade_time[0]-1000]
        
        f_sharpes, f_sharpe_series = sharpe_agent_group(agent_signals=f_order_action,
                               agent_shares=f_order_shares,
                               agent_prices=f_order_prices,
                               prices=prices, sharpe_series=True)
        c_sharpes, c_sharpe_series = sharpe_agent_group(agent_signals=c_order_action,
                                    agent_shares=c_order_shares,
                                    agent_prices=c_order_prices,
                                    prices=prices, sharpe_series=True)
        s_sharpes, ret = sharpe_one_agent(agent_signal=s_signal,
                                        agent_shares=s_shares,
                                        agent_prices=s_prices,
                                        agent_action_time=s_action_time,
                                        prices=prices, ret=True, evaluate_step=1)
        
        # save data
        f_sharpe_save.append(f_sharpes)
        c_sharpe_save.append(c_sharpes)
        s_sharpe_save.append(s_sharpes)
    except:
        pass

cwd = os.path.join(os.getcwd(), 'spoofing_simulation', 'sharpe_ratio')
f_sharpe_save = np.save(os.path.join(cwd, 'f_sharpe_save.npy'), np.array(f_sharpe_save))
c_sharpe_save = np.save(os.path.join(cwd, 'c_sharpe_save.npy'), np.array(c_sharpe_save))
s_sharpe_save = np.save(os.path.join(cwd, 's_sharpe_save.npy'), np.array(s_sharpe_save))
