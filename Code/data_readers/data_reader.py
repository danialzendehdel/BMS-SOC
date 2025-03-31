import scipy.io as sio
from scipy.io import loadmat, matlab
from scipy.io.matlab import mat_struct
import numpy as np
import pandas as pd



def load_matlab_data(file_path):
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    matlab_struct = data['data']

    # Create a dictionary to store all fields
    fields_dict = {}

    # Automatically store all fields in the dictionary
    for field in matlab_struct._fieldnames:
        field_value = getattr(matlab_struct, field)
        
        # Check if it's a MATLAB struct with sub-fields
        if isinstance(field_value, mat_struct):
            # print("Sub-fields:", field_value._fieldnames)
            for field_sub in field_value._fieldnames:
                fields_dict[field_sub] = getattr(field_value, field_sub)
        else:
            fields_dict[field] = field_value

    # Now you can access any field like: fields_dict['OCV'], fields_dict['SOC'], etc.
    # print("\nAvailable fields:", fields_dict.keys())
    # for key_ in fields_dict.keys():
    #     if isinstance(fields_dict[key_], np.ndarray):
    #         print(f"{key_}: shape {fields_dict[key_].shape}, type {type(fields_dict[key_])}")
    #     elif isinstance(fields_dict[key_], mat_struct):
    #         print(f"{key_}: MATLAB struct with fields {fields_dict[key_]._fieldnames}")
    #     else:
    #         print(f"{key_}: type {type(fields_dict[key_])}")

    # Example: Create DataFrame with OCV and SOC
    # df = pd.DataFrame({'OCV': fields_dict['OCV'], 'SOC': fields_dict['SOC']})
    # df = pd.DataFrame({'R': fields_dict['R']})
    # print("\nDataFrame:")
    # print(df)

    # print(fields_dict['R'])

    return fields_dict


def load_matlab_data2(file_path):
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    matlab_struct = data["out_data"]

    fields_dict = {}
    for field in matlab_struct._fieldnames:
        field_value = getattr(matlab_struct, field)

        # Check if it's a MATLAB struct with sub-fields
        if isinstance(field_value, mat_struct):
            # print("Sub-fields:", field_value._fieldnames)
            for field_sub in field_value._fieldnames:
                fields_dict[field_sub] = getattr(field_value, field_sub)
        else:
            fields_dict[field] = field_value


    I = fields_dict['I']
    P = fields_dict['P']
    SoC = fields_dict['SOC']

    return I, P, SoC

if __name__ == "__main__":
    load_matlab_data2("/home/danial/Documents/Codes/BMS-SOC/Code/data/cycle.mat")
# load_matlab_data()