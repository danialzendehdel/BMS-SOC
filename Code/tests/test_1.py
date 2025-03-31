import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configuration.params_config import load_full_config
import matplotlib.pyplot as plt



def _sfunct(soc, c_rate, temperature, params):
        """
        Calculate stress factor based on SOC and C-rate
        Args:
            c_rate: Current C-rate (current/nominal_capacity)
            temperature: Cell temperature in Celsius
        """
        aging_model_config = params.aging_model
        # Get coefficients based on SOC
        a, b = (aging_model_config.case_b_45 
                if soc >= 45 
                else aging_model_config.case_l_45)
        
        # Calculate stress factor
        # C_rate: Current rate normalized to battery charge capacity
        stress = (a * soc + b) * np.exp(
            -(aging_model_config.Ea_J_per_mol + aging_model_config.h * c_rate) / 
            (aging_model_config.Rg_J_per_molK * (273.15 + temperature))
        )
        
        return stress


def _coupled_battery_degradation(current, soc, charge_throughput, nominal_ah, SOH, c_rate, params):
    time_step = 0.25 

    # Charge throughput increment (Ah)
    delta_ah = abs(current) * time_step  # Ah
    charge_throughput += delta_ah

    # Stress factor (using your existing sfunct)
    s_value = _sfunct(soc, c_rate, temperature=params.aging_model.constant_temperature, params=params)
    s_value = max(s_value, 1e-10)  # Prevent division by zero

    # Eq. 5 Ah_total
    ah_total = (20.0 / s_value) ** (1.0 / params.aging_model.exponent_z)

    C_use = max(SOH * nominal_ah, 1e-10)  # Prevent division by zero
    N_val = max(ah_total / C_use, 1e-10)  # Prevent division by zero

    # delta_soh = - ( |I|*dt ) / [2 * N_val * C_use ]
    delta_soh = - (abs(current) * time_step) / (2 * N_val * C_use)
    SOH += delta_soh
    SOH = max(0.0, min(SOH, 1.0))

    dSOC = - (current * time_step) / C_use * 100.0
    soc += dSOC
    # soc = np.clip(soc, 20, 80)

    # Track Q_loss_percent
    Q_loss_percent = (1.0 - SOH) * 100.0

    return soc, SOH, ah_total, charge_throughput, Q_loss_percent


def simulation():
    c_rate = 2.82
    soc_avg = 38.5 
    temp = 25
    nominal_ah = 112.7
    current = c_rate * nominal_ah
    soh = 1 
    charge_throughput = 0
    config_file = 'Code/configuration/params_2.yml'
    params = load_full_config(config_file)

    n_cycles = 5
    soc_value = 20
    info = {
        'soc': [],
        'soh': [],
        'ah_total': [],
        'charge_throughput': [],
        'Q_loss_percent': []
    }

    for i in range(n_cycles):
         
        while 20 < soc_value < 57:
            current = c_rate * nominal_ah
            soc_value, soh, ah_total, charge_throughput, Q_loss_percent = _coupled_battery_degradation(current, soc_value, charge_throughput, nominal_ah, soh, c_rate, params)
            info['soc'].append(soc_value)
            info['soh'].append(soh)
            info['ah_total'].append(ah_total)
            info['charge_throughput'].append(charge_throughput)
            info['Q_loss_percent'].append(Q_loss_percent)

            print(f"End of charge phase: SOH={soh:.4f}, SOC={soc_value:.1f}%")
            

        while soc_value > 20: 
            current = -c_rate * nominal_ah
            soc_value, soh, ah_total, charge_throughput, Q_loss_percent = _coupled_battery_degradation(current, soc_value, charge_throughput, nominal_ah, soh, c_rate, params)
            info['soc'].append(soc_value)
            info['soh'].append(soh)
            info['ah_total'].append(ah_total)
            info['charge_throughput'].append(charge_throughput)
            info['Q_loss_percent'].append(Q_loss_percent)

        print(i)
        i += 1

    return info
        
        
        

def generate_soc_profile(
    start_soc=20.0,   # Starting SoC in percent
    high_soc=57.0,    # High SoC target in percent
    n_cycles=5,       # Number of full cycles
    steps_per_phase=100
):
    """
    Generates a piecewise-linear SoC profile where each cycle goes:
      start_soc -> high_soc -> start_soc
    repeated for n_cycles.

    Args:
      start_soc     : SoC at the beginning of each cycle low point (30% default).
      high_soc      : SoC target at the top of each charge (70% default).
      n_cycles      : how many up/down cycles to produce (default 5).
      steps_per_phase : how many discrete steps to use per charge or discharge phase.

    Returns:
      time_array: 1D array of time indices
      soc_array : 1D array of SoC values in [%, %]
    """

    # We'll have 2 phases per cycle (charge up, discharge down).
    total_phases = 2 * n_cycles

    # We'll store SoC in a list; time in a list
    soc_list = []
    time_list = []

    # Keep a running index for time
    t = 0

    # For each cycle:
    for cycle_index in range(n_cycles):
        # Phase 1: charge from start_soc -> high_soc
        # We'll create a linear ramp from e.g. 30% -> 70% in 'steps_per_phase' increments
        soc_up = np.linspace(start_soc, high_soc, steps_per_phase, endpoint=False)
        for soc_val in soc_up:
            soc_list.append(soc_val)
            time_list.append(t)
            t += 1  # increment time

        # Phase 2: discharge from high_soc -> start_soc
        soc_down = np.linspace(high_soc, start_soc, steps_per_phase, endpoint=False)
        for soc_val in soc_down:
            soc_list.append(soc_val)
            time_list.append(t)
            t += 1

    # Finally, append the last end-of-discharge point to close the cycle
    soc_list.append(start_soc)
    time_list.append(t)

    return np.array(time_list), np.array(soc_list)



# if __name__ == "__main__":
#     # Generate a SoC profile for 5 cycles from 30%->70%->30%, 20 steps each phase
#     t_array, soc_array = generate_soc_profile(
#         start_soc=30.0,
#         high_soc=70.0,
#         n_cycles=5,
#         steps_per_phase=20
#     )

#     # Plot the result
#     plt.figure(figsize=(8,4))
#     plt.plot(t_array, soc_array, marker='o', label="SoC profile")
#     plt.title("SoC cycles: 30% to 70% and back to 30%, repeated 5 times")
#     plt.xlabel("Time (arbitrary steps)")
#     plt.ylabel("SoC (%)")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#     plt.savefig('soc_profile.png')

if __name__ == '__main__':
    t_array, soc_array = generate_soc_profile()
    info = simulation()

    plt.figure(figsize=(8,4))
    plt.plot(info['soc'], info['soh'], marker='o', label="SoC profile")
    plt.title("SoC cycles: 30% to 70% and back to 30%, repeated 5 times")
    plt.xlabel("Time (arbitrary steps)")
    plt.ylabel("SoC (%)")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('soc_profile.png')
    # simulation()



