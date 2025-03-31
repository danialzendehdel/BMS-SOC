import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Code.data_readers.load_data import matlab_load_data, PNetDataLoader
from Code.data_readers.data_reader import load_matlab_data2
from optimizers.optimize_2 import OptimizeSoC
import yaml
from scipy.interpolate import interp1d
from Code.utils.utils import *
from tqdm import tqdm
from Code.utils.journal import Journal

def mse_(actual, pred):
    
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean()


def run(configuration_path):

    journaling = Journal("results1", "test")

    try:
        with open(configuration_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration{e}")
        return None

    # data
    R_lookup, SOC_lookup, C_rate_lookup, OCV_lookup = matlab_load_data(config["data"]["matlab_data"])
    I_gt, P_net, SoC_gt = load_matlab_data2(config["data"]["matlab_data_test"])

    mask = (SoC_gt >= 20) & (SoC_gt <= 80)
    SoC_gt = SoC_gt[mask]
    P_net = P_net[mask]
    I_gt = I_gt[mask]/1000 # A


    variables = {
    "R_lookup": R_lookup,
    "SOC_lookup": SOC_lookup,
    "C_rate_lookup": C_rate_lookup,
    "OCV_lookup": OCV_lookup,
    "I_gt": I_gt,
    "P_net": P_net,
    "SoC_gt": SoC_gt
    }

    for var_name, var_data in variables.items():
        data_statistics(var_data, var_name)
        journaling._get_datainfo(var_data, var_name)
    


    # Configurations
    soc_init = config["battery"]["soc_init"]
    epsilon = config["train"]["epsilon"]
    nominal_capacity = config["battery"]["nominal_capacity"]
    nominal_current = config["battery"]["nominal_current"]
    time_interval = config["train"]["time_interval"]/3600 # minutes
    nominal_Ah = config["battery"]["nominal_Ah"]
    ga = config["train"]["GA"]
    print(ga)

    # results: info from the optimization
    info_f = {
            "c_rate": [],
            "R": [],
            "current": [],
            "error_ga": [],
            "soc": [],
            "MSE_soc": [],
            "MSE_current":[],
        }
    # info_f["soc"].append(soc_init)
    soc = soc_init

    # P_net = P_net[136597:-1]


    optimizer = OptimizeSoC(
                            SOC_lookup,
                            C_rate_lookup,
                            R_lookup,
                            OCV_lookup,
                            epsilon,
                            nominal_current,
                            nominal_capacity,
                            GA=ga,
                            verbose=False)

    for ind in tqdm(range(len(P_net))): # power
        # print(f"power index: {ind}")
        power = P_net[ind] # Net load

        if power == 0:
            journaling._get_warning(e="power is zero")
            # print("Power is zero")
            
            soc = soc
            mse_v = mse_(SoC_gt[ind], soc)
            mse_I = mse_(I_gt[ind], 0)

            info_f["c_rate"].append(0.0)
            info_f["R"].append(0.0)
            info_f["current"].append(0.0)
            info_f["error_ga"].append(0.0)
            info_f["soc"].append(soc)
            info_f["MSE_soc"].append(mse_v)
            info_f["MSE_current"].append(mse_I)
            continue
        # Todo interpolate the OCV
        # index = np.abs(SOC_lookup - soc_init).argmin()

        # OCV interpolation
        # ocv = interp1d(SOC_lookup, OCV_lookup, kind='linear', fill_value="extrapolate")(soc)

        # optimizer = OptimizeSoC(soc,
        #                         SOC_lookup,
        #                         C_rate_lookup,
        #                         R_lookup,
        #                         ocv,
        #                         power,
        #                         epsilon,
        #                         nominal_current,
        #                         nominal_capacity,
        #                         GA=ga,
        #                         verbose=False)


        new_R, new_I, info = optimizer._optimize(soc, power)
        soc +=  new_I * time_interval/ nominal_Ah * 100
        
        mse_soc = mse_(SoC_gt[ind], soc)
        mse_I = mse_(I_gt[ind], new_I)

        # Check the SoC
        if soc > 80 or soc < 20:
            print(f"Warning: SoC is not in the range: {soc}")

        info_f["c_rate"].append(info["c_rate"])
        info_f["R"].append(info["R"])
        info_f["current"].append(info["current"][-1])
        info_f["error_ga"].append(info["error_ga"])
        info_f["soc"].append(soc)
        info_f["MSE_soc"].append(mse_soc)
        info_f["MSE_current"].append(mse_I)
        

    print(type(info_f['current']))

    save_csv(info_f, ga, "results_GA.csv")
    plot_log(I_gt, P_net, SoC_gt, info_f, GA=ga)

    mse_soc_all = mse_(SoC_gt[:ind+1], info_f["soc"])
    mse_I_all   = mse_(I_gt[:ind+1],   info_f["current"])

    print("Overall SoC MSE:", mse_soc_all)
    print("Overall current MSE:", mse_I_all)


    return info_f




if __name__ == "__main__":
    kir = run("configuration/params.yml")
    print((len(kir["current"])))
    print(len(kir["error_ga"]))
    print(len(kir["R"]))
