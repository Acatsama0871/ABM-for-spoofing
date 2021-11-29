import os
import queue
import concurrent.futures
import numpy as np
from tqdm import tqdm
from model import SimulationModel

# run model function
def run_model():
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

    for _ in range(num_sim_times):
        model.step()
    
    return model.price_series, model.get_return_series(), np.array(model.agent_groups[1].agents_traded_action_record), np.array(model.agent_groups[1].agents_traded_price_record), np.array(model.agent_groups[2].agents_traded_action_record), np.array(model.agent_groups[2].agents_traded_price_record)

if __name__ == "__main__":
    # set path
    cwd = os.path.join(os.getcwd(), 'spoofing_simulation', 'detrimental_effect')

    # run model function
    price_series = queue.Queue()
    ret_series = queue.Queue()
    f_actions_series = queue.Queue()
    f_order_prices_series = queue.Queue()
    c_actions_series = queue.Queue()
    c_order_prices_series = queue.Queue()
    num_MC = 100
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        with tqdm(total=num_MC) as progress:
            results = []
            for _ in range(num_MC):
                future = pool.submit(run_model)
                future.add_done_callback(lambda _: progress.update())
                results.append(future)
            
            for cur_result in concurrent.futures.as_completed(results):
                cur_price_series, cur_ret_series, cur_f_action, cur_f_order_price, cur_c_action, cur_c_order_price = cur_result.result()
                price_series.put(cur_price_series)
                ret_series.put(cur_ret_series)
                f_actions_series.put(cur_f_action)
                f_order_prices_series.put(cur_f_order_price)
                c_actions_series.put(cur_c_action)
                c_order_prices_series.put(cur_c_order_price)
    
    price_series = np.array(price_series.queue)
    ret_series = np.array(ret_series.queue)
    f_actions_series = np.array(f_actions_series.queue)
    f_order_prices_series = np.array(f_order_prices_series.queue)
    c_actions_series = np.array(c_actions_series.queue)
    c_order_prices_series = np.array(c_order_prices_series.queue)
    np.save(os.path.join(cwd, 'prices.npy'), price_series)
    np.save(os.path.join(cwd, 'returns.npy'), ret_series)
    np.save(os.path.join(cwd, 'f_actions_series.npy'), f_actions_series)
    np.save(os.path.join(cwd, 'c_actions_series.npy'), c_actions_series)
    np.save(os.path.join(cwd, 'f_order_prices_series.npy'), f_order_prices_series)
    np.save(os.path.join(cwd, 'c_order_prices_series.npy'), c_order_prices_series)
