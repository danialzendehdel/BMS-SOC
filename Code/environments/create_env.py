import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.BMS_env_coupled import BMSEnv
from configuration.params_config import load_full_config
from data_readers.data_handler import DataHandler
from utils.journal import Journal
from optimizers.optimize_2 import OptimizeSoC 
from data_readers.load_data import matlab_load_data
from utils import journal

def create_env(path, journalist_ins):

    # journalist = Journal("Code/results/Summary", "test")
    journalist = journalist_ins
    params = load_full_config(path)
    data_handler = DataHandler(params.data.P_net)
    journalist._get_data_stats(data_handler.stats)
    journalist._plot_energy(data_handler.df)

    R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = matlab_load_data(params.data.matlab_data)

    variables = {
    "R_lookup": R_lookup,
    "SOC_lookup": SOC_lookup,
    "C_rate_lookup": C_rate_lookup,
    "OCV_lookup": OCV_lookup,
    }
    for var_name, var_data in variables.items():
        journalist._get_datainfo(var_data, var_name)
    # adjust the R_lookup table and OCV_lookup table # TODO: what about the SoC lookup
    # R_lookup *= (params.environment.battery.Ns / params.environment.battery.Np)
    # OCV_lookup *= params.environment.battery.Ns

    optimizer = OptimizeSoC(
                            SOC_lookup,
                            C_rate_lookup,
                            R_lookup,
                            OCV_lookup,
                            params.train.epsilon,
                            nominal_current=params.environment.battery.nominal_current,
                            nominal_capacity=params.environment.battery.nominal_capacity,
                            params=params,
                            journalist_ins=journalist,
                            GA=params.train.GA,
                            verbose=False)

    
    journalist._get_data_stats(data_handler.stats)

    variables = {
    "R_lookup": R_lookup,
    "SOC_lookup": SOC_lookup,
    "C_rate_lookup": C_rate_lookup,
    "OCV_lookup": OCV_lookup,
    }
    for var_name, var_data in variables.items():
        journalist._get_datainfo(var_data, var_name)


    configuration_values = {
        "load_min": data_handler.stats.get("load_min"),
        "load_max": data_handler.stats.get("load_max"),
        "solar_max": data_handler.stats.get("solar_max"),
        "solar_min": data_handler.stats.get("solar_min")
    }

    env = BMSEnv(data_handler=data_handler,
                 current_optimizer=optimizer,
                 params_ins=params,
                 journalist=journalist,
                 verbose=True ,
                 **configuration_values
                 )
    
    return env




if __name__ == "__main__":
    journalist = Journal("Code/results/Training", "training")
    env = create_env('Code/configuration/params_2.yml', journalist)
    print(f"Observation space dimension: {env.observation_space.shape[0]}")
    print(f"Action space: {env.action_space}")

    # Initialize the environment first
    # obs, info = env.reset()
    # print("\nInitial observation after reset:", obs)

    # print(env.observation_space)
    # print(env.observation_space.low)
    # print(env.observation_space.high)


    # for i in range(10):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     # print(env.info.items())
    #     # print(f"\nStep {i+1}:")
    #     # print(f"Action: {action}")
    #     # print(f"Observation: {obs}")
    #     # print(f"Reward: {reward}")
    #     # if terminated or truncated:
    #     #     obs, info = env.reset()
    
    for key, value in env.__dict__.items():
        print(f"{key}: {value}")