import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import datetime



def save_csv(data, GA, name:str) -> None:

    path = "/home/danial/Documents/Codes/BMS-SOC/Code/results/csv"
    if not os.path.exists(path):
        os.mkdir(path)

    if not GA:
        name = "Deterministic_"
    else:
        name = "GA_"

    print("Length of soc:", len(data["soc"]))
    print("Length of current:", len(data["current"]))

    # with open(os.path.join(path, name + ".csv"), 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Parameter", "Value"])
    #     for key, value in data.items():
    #         writer.writerow([key, value])

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')


    df = pd.DataFrame(data, columns=["soc", "current"])  # optional column names
    df.to_csv(os.path.join(path, name + formatted_time + ".csv"), index=False)

def plot_log(I_gt, P_gt, SoC_gt, info, GA=False):

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')

    saving_dir = "/home/danial/Documents/Codes/BMS-SOC/Code/results/figures"
    if not GA:
        prefix = "Deterministic_"
    else:
        prefix = "GA_"

    # ground truth

    fig, ax = plt.subplots(3, 1, figsize=(15, 5))
    ###
    #ax[0].plot(I_gt, label='Ground Truth')
    #ax[0].set_title('Current')
    #ax[0].set_xlabel('Time')
    #ax[0].set_ylabel('Current')
    #ax[0].legend()

    ax[0].plot(P_gt, label='Ground Truth')
    ax[0].set_title('Power')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Power')
    ax[0].legend()
    

    # plt.savefig(os.path.join(saving_dir, prefix + 'ground_truth.png'), dpi=300)

    #plt.close()

    
    #fig, ax = plt.subplots(2, 1, figsize=(15, 5))
    ax[1].plot(SoC_gt[:len(info['soc'])], label='Ground Truth')
    ax[1].plot(info['soc'], label='Predicted')
    ax[1].set_title('SoC')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('SoC')
    ax[1].set_ylim([0, 1.1 * max(max(SoC_gt[:len(info['soc'])]), max(info['soc'][:len(info['soc'])]))])
    ax[1].legend()

    ax[2].plot(I_gt[:len(info['current'])], label='Ground Truth')
    ax[2].plot(info['current'], label='Predicted')
    ax[2].set_title('Current')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Current')
    ax[2].legend()

    plt.savefig(os.path.join(saving_dir, prefix + formatted_time+  'predicted.png'), dpi=300)

    plt.close()

    fig, ax = plt.subplots(2,1, figsize=(15,5))

    ax[0].plot(info["MSE_soc"], label="MSE_SoC")
    ax[0].set_title("MSE SoC")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("MSE SoC")
    ax[0].legend()

    ax[1].plot(info["MSE_current"], label="MSE_I")
    ax[1].set_title("MSE I")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("MSE I")
    ax[1].legend()

    plt.savefig(os.path.join(saving_dir, prefix + formatted_time + 'MSE.png'), dpi=300)
    plt.close()

def data_statistics(data, name:str) -> None:

    box_width = 70
    separator = "=" * box_width
    def format_line(content:str) -> str:
        return f"|{content.ljust(box_width - 4)}|"

    print(separator)

    print(format_line("Data Statistics " + name))
    print(format_line(f"length: {len(data)}"))
    print(format_line(f"max: {np.max(data)}"))
    print(format_line(f"min: {np.min(data)}"))
    print(format_line(f"mean: {np.mean(data)}"))
    print(format_line(f"std: {np.std(data)}"))
    print(separator)
    print("\n")

    return None




