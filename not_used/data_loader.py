import numpy as np
import os
from Code.data_readers.data_reader import load_matlab_data
import pandas as pd

class DataLoader:
    def __init__(self, net_load_path, values_path):

        if not os.path.exists(net_load_path):
            assert False, f"File {net_load_path} does not exist"

        if not os.path.exists(values_path):
            assert False, f"File {values_path} does not exist"

        params = load_matlab_data(values_path)
        self.R_v, self.SOC_v, self.C_rate_v = params['R'], params['SOC'], params['C_rate']

        loads_v = pd.read_csv(net_load_path)

        self.loads_v = loads_v["load"]
        self.solar_v = loads_v["solar"]

    def __getitem__(self, index):
        return self.SOC_v[index]

    def __len__(self):
        return len(self.R_v)


if __name__ == '__main__':
    Net_load_path = "/home/danial/Documents/Codes/BMS-SOC/Code/data-n/processed_data_661.csv"
    data_path = "/home/danial/Documents/Codes/BMS-SOC/Code/data/NMC_cell_data.mat"

    dl = DataLoader(Net_load_path, data_path)

    print(len(dl))