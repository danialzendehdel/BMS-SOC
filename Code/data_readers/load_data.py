import os
from Code.data_readers.data_reader import load_matlab_data
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd


def matlab_load_data(data_path):
    
    # Net_load_path = "/home/danial/Documents/Codes/BMS-SOC/Code/data-n/processed_data_661.csv"
    # data_path = "/home/danial/Documents/Codes/BMS-SOC/Code/data/NMC_cell_data.mat"
    
    if not os.path.exists(data_path):
        assert False, f"File {data_path} does not exist"


    params = load_matlab_data(data_path)
    R_values = params['R']
    # SoC_values = np.expand_dims(['SOC'],1)
    # C_rate_values = np.expand_dims(['C_rate'],1)
    soc_values = params['SOC']
    c_rate_values = params['C_rate']
    OCV_values =  params['OCV']
    # print(len(OCV_values))



    return R_values, soc_values, c_rate_values, np.flip(OCV_values)

class ParamsDataLoader:
    def __init__(self, path):
        if not os.path.exists(path):
            assert False, f"File {path} does not exist"

        self.R_values, self.SOC_values, self.C_rate_values, self.OCV_values = matlab_load_data(path)

        self.current_index = 0


    def __getitem__(self, index):

        return self.R_values[index], self.SOC_values[index], self.C_rate_values[index], self.OCV_values[index]




class PNetDataLoader:
    def __init__(self, path):

        if not os.path.exists(path):
            assert False,f"File {path} does not exist"

        df = pd.read_csv(path)

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        df.dropna(inplace=True)
        df = df.reset_index()
        df = df.sort_values("datetime").drop_duplicates("datetime")
        df["solar"] = df["solar"].clip(lower=0)
        df["P"] = ((df["load"] - df["solar"]) / (8.5) )* 18 / 20  # x kw/ 8.5 kw * 18

        self.df = df
        self.current_index = 0
        print(f"max: {df["P"].max()}")
        print(f"min: {df["P"].min()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
            return self.df["P"].iloc[index]


    
    
if __name__ == "__main__":
    x,y,z,m = matlab_load_data("/home/danial/Documents/Codes/BMS-SOC/Code/data/NMC_cell_data.mat")
    indices = np.argwhere(y == 50).flatten()
    print(indices)
    print(m[indices][0])
    # P_net = PNetDataLoader("/home/danial/Documents/Codes/BMS-SOC/Code/data-n/processed_data_661.csv")
    # params = ParamsDataLoader("/home/danial/Documents/Codes/BMS-SOC/Code/data/NMC_cell_data.mat")
    #
    # index_ = 10
    #
    # print(f"x : {P_net[index_]}")
    # print(params[index_])
    # data_path = None
    # x,y,z = matlab_load_data(data_path)
    #
    # print(x.shape)
    # print(y.shape)
    # print(z.shape)
    #
    # interpolator = RegularGridInterpolator((y, z), x, method='linear')
    #
    # soc_query = 5  # Example query point for SOC
    # c_rate_query = 0.5  # Example query point for C-rate
    #
    # print("SOC range:", min(y), "to", max(y))
    # print("C-rate range:", min(z), "to", max(z))
    #
    # if soc_query < min(y) or soc_query > max(y) or \
    #         c_rate_query < min(z) or c_rate_query > max(z):
    #     print("Query point is out of bounds!")
    # else:
    #     points_to_interpolate = np.array([[soc_query, c_rate_query]])
    #     interpolated_R_values = interpolator(points_to_interpolate)
    #     print("Interpolated R value:", interpolated_R_values)
    #
    # # Perform interpolation
    # interpolated_R_values = interpolator(points_to_interpolate)
    # print("Interpolated R value:", interpolated_R_values)


