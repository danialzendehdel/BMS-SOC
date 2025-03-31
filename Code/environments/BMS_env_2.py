from typing import Any
import gymnasium as gym 
from gymnasium import spaces
import numpy as np 



# TODO : charge and discharge rate 
"""
    Action, P_batt: The agent decide on the amount of power to charge/discharge (-/+)
    observation: the net load (P_l - P_g), state of charge (SoC) and Time of day and weekday
    Reward: Q_loss and P_loss

    End of Episode (EOL): when 20% of capacity fade 

    - The price of energy is constant 
"""


class BMSEnv(gym.Env):
    """
    data_handler: provides solar and load power information to calculate Net load. 
    current_optimizer: instance of current and resistance optimizer 
    params_ins: all constant information regarding the battery and hyper parameters
    journalist: an instance of Journal class, saves all warnings and error in a plain text 
    """
    def __init__(self, data_handler, current_optimizer, params_ins, journalist, verbose=True, **kwargs):
        super(BMSEnv, self).__init__()
        
        # TODO: what should be the time metric s or m , or hour
        self.params = params_ins
        self.versbose = verbose

        self.soc = params_ins.environment.battery.soc_init
        self.time_step = params_ins.train.time_interval
        self.nominal_ah = params_ins.environment.battery.nominal_Ah

        # RL Coefficients 
        self.coeff_q_loss = params_ins.environment.reward_coeff.coeff_q_loss
        self.coeff_p_loss = params_ins.environment.reward_coeff.coeff_p_loss

        self.current_optimizer = current_optimizer
        self.data_handler = data_handler
        self.journalist = journalist
        self.info = self._getinfo()
        self.episode_length = 0 

        self.aging_model_config = params_ins.aging_model
        self.charge_throughput = 0
        self.Q_loss = self.nominal_ah # 100% starts at 0% fade # nominal_ah 
        self.Q_loss_percent = 100 # 100%
        self.Q_loss_EOL = params_ins.aging_model.q_loss_eol

        self.load_min,  self.load_max = kwargs.get("load_min"), kwargs.get("load_max")
        self.solar_min, self.solar_max = kwargs.get("solar_min"), kwargs.get("solar_max")


       
        # soc , net_power
        self.observation_space = spaces.Box(low=np.array([self.params.environment.battery.soc_min, 
                                                          self.load_min,
                                                          self.solar_min]), 
                                            high=np.array([self.params.environment.battery.soc_max,
                                                           self.load_max,
                                                           self.solar_max]),
                                            dtype=np.float64)
        

        # TODO: action is charging rate or amount of power has discharged
        self.action_space = spaces.Box(low=np.array([self.params.environment.battery.max_discharge_rate_kW]),
                                       high=np.array([self.params.environment.battery.max_charge_rate_kW]),
                                       dtype=np.float64)
        



    def _getinfo(self):
        return  {
            "soc_value" : [],
            "soc_violation": [],
            "soc_clipped": [],
            "current": [],
            "resistance": [],
            "action_value": [],
            "action_violation": [],
            "action_clipped": [],
            "p_loss": [],
            "q_loss": [],
            "q_loss_percent": [],
            "throughput_charge": [],
            "q_loss_cumulative": [],
            "reward": []
        }
     

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.episode_length = 0
        self.soc = self.params.environment.battery.soc_init
        self.Q_loss = self.nominal_ah
        self.Q_loss_percent = 100
        self.charge_throughput = 0 

        initial_state = self.data_handler[self.episode_length]
        self.p_l = initial_state["load"]
        self.p_g = initial_state["solar"]

        # for key in self.info:
        #     self.info[key] = []
        self.info = self._getinfo()

        self.journalist._process_smoothly("RESTARTED")

        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([
            float(self.soc),
            float(self.p_l),
            float(self.p_g)
        ], dtype=np.float32)


    def _get_soc(self, action):
        resistance, current, _ = self.current_optimizer._optimize(self.soc, action * 1000)  # (current soc, P_batt)
        
        # Current should already have the correct sign from the optimizer
        soc_new = self.soc + current * self.time_step / self.nominal_ah * 100

        # Check the new soc value to be in boundary 
        if np.isnan(soc_new):
            self.journalist._get_error("The new SoC is NaN")
            raise ValueError("The new SoC: {soc_new} is NaN")
        
        soc_clipped = np.clip(soc_new, self.observation_space.low[0], 
                              self.observation_space.high[0])
        delta_soc = abs(float(soc_clipped) - float(self.soc))
        soc_boundary_violation = not np.isclose(soc_clipped, soc_new, atol=1e-4)

        # update the soc
        self.soc = soc_clipped
        
        self.info["soc_value"].append(soc_new)
        self.info["soc_violation"].append(soc_boundary_violation)
        self.info["soc_clipped"].append(soc_clipped)
        self.info["current"].append(current)
        self.info["resistance"].append(resistance)

        # Debugging
        self.journalist._get_warning(
            f"SoC boundary violation: {self.observation_space.low[0]} < {self.soc} < {self.observation_space.high[0]}"
            )  if soc_boundary_violation else None

        return soc_clipped, delta_soc, current, resistance




    
    def _get_reward(self, current, resistance):
        
        p_loss_step, q_loss_step = self._battery_degradation(resistance=resistance, current=current)
        step_reward = -(p_loss_step * self.coeff_p_loss + q_loss_step * self.coeff_q_loss)

        self.info["reward"].append(step_reward)

        return step_reward




    def step(self, action):
        self.episode_length += 1

        # Add debug print to check episode length
        if self.episode_length >= 9995:  # Add this debug line
            print(f"Episode length: {self.episode_length}")
            print(f"Data handler length: {len(self.data_handler)}")
            print(f"Dataset complete: {self.episode_length >= len(self.data_handler)}")

        # Action boundary check Debugging of action
        action_bounded, delta_action = self._check_action(action)

        # update SoC, get delta SoC, get current resistance
        soc_bounded, delta_soc, current_step, resistance_step = self._get_soc(action_bounded)

        reward_step = self._get_reward(current=current_step, resistance=resistance_step)

        # TODO

        if self.episode_length < len(self.data_handler):
            next_state = self.data_handler[self.episode_length]
            self.p_l = next_state["load"]
            self.p_g = next_state["solar"]
            dataset_complete = False
        else:
            dataset_complete = True

        obs = self._get_obs()
        wasted = (self.Q_loss_percent <= self.Q_loss_EOL)

        if  wasted or dataset_complete:
            done = True
        else:
            done = False 

        truncated = False   
        info_dict = self.info

        
        self._get_steps_printed() if self.versbose else None

        return obs, reward_step, done, truncated, info_dict




    def _check_action(self, action):

        # if isinstance(action, np.ndarray):
        action = float(action)
        assert isinstance(action, float), f"Action should be float, got {type(action)}"

        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)
        delta_action = abs(action_clipped - action)
        action_boundary_violation = not np.isclose(action_clipped, action, atol=1e-4)

        # info 
        self.info["action_value"].append(action)
        self.info["action_violation"].append(action_boundary_violation)
        self.info["action_clipped"].append(action_clipped)

        # Debugging
        self.journalist._get_warning(
            f"Action boundary violation: {self.action_space.low[0]} < {action} < {self.action_space.high[0]}"
            )  if action_boundary_violation else None

        return action_clipped, delta_action


    
    def _battery_degradation(self, resistance, current):
        # Add unit checks
        assert 0.01 <= resistance <= 1.0, f"Resistance {resistance}Ω outside expected range"
        assert -200 <= current <= 200, f"Current {current}A outside expected range"
        assert 0 <= self.soc <= 100, f"SOC {self.soc}% outside valid range"
        
        # Power loss calculation (W = Ω * A²)
        p_loss_step = resistance * np.pow(current, 2) # Watt
        
        # Update accumulated charge throughput (Ah)
        delta_ah = abs(current) * self.time_step  # Ampere-hours for this step
        self.charge_throughput += delta_ah  # Accumulate total charge throughput
        
        # Calculate C-rate
        c_rate = abs(current) / self.nominal_ah  # h⁻¹
        
        # Calculate stress factor
        s_value = self._sfunct(c_rate, temperature=self.aging_model_config.constant_temperature)
        
        # Calculate capacity loss using accumulated throughput
        # From the paper: Q_loss = s(c-rate, T, SOC) * (Ah_throughput)^z
        q_loss_step = s_value * (self.charge_throughput ** self.aging_model_config.exponent_z)
        
        # Calculate incremental capacity loss
        delta_q_loss = q_loss_step - (s_value * ((self.charge_throughput - delta_ah) ** self.aging_model_config.exponent_z))
        
        self.journalist._get_warning(f"q_loss_step is negative: {delta_q_loss}") if delta_q_loss < 0 else None
        
        # Update total capacity loss
        self.Q_loss -= delta_q_loss
        self.Q_loss_percent = (self.Q_loss/self.nominal_ah) * 100

        # Store info
        self.info["p_loss"].append(p_loss_step)
        self.info["q_loss"].append(delta_q_loss)
        self.info["q_loss_percent"].append(self.Q_loss_percent)
        self.info["throughput_charge"].append(self.charge_throughput)
        self.info["q_loss_cumulative"].append(self.Q_loss)

        # Add debug prints every 100 steps
        if self.episode_length % 100 == 0:
            self.journalist._process_smoothly(
                f"Step {self.episode_length}: "
                f"Capacity: {self.Q_loss_percent:.2f}%, "
                f"Total Throughput: {self.charge_throughput:.2f} Ah, "
                f"C-rate: {c_rate:.2f}"
            )

        return p_loss_step, delta_q_loss


    
    def _sfunct(self, c_rate, temperature):
        """
        Calculate stress factor based on SOC and C-rate
        Args:
            c_rate: Current C-rate (current/nominal_capacity)
            temperature: Cell temperature in Celsius
        """
        # Get coefficients based on SOC
        a, b = (self.aging_model_config.case_b_45 
                if self.soc >= 45 
                else self.aging_model_config.case_l_45)
        
        # Calculate stress factor
        # C_rate: Current rate normalized to battery charge capacity
        stress = (a * self.soc + b) * np.exp(
            -(self.aging_model_config.Ea_J_per_mol + self.aging_model_config.h * c_rate) / 
            (self.aging_model_config.Rg_J_per_molK * (273.15 + temperature))
        )
        
        return stress


    def _get_steps_printed(self):

        box_width = 90
        separator = "=" * box_width
        
        def format_line(content: str) -> str:
            """Formats a single line within the box with padding."""
            return f"| {content.ljust(box_width - 4)} |"
        
        def format_header(title: str) -> str:
            """Formats a header line centered within the box, surrounded by '='."""
            return f"|{title.center(box_width - 2, '-')}|"
        
        
        print(separator)
        
        # Step Information

        print(format_line(f" Episode length: {self.episode_length}"))
        # current
        print(format_line(f" current: {self.info["current"][-1]}"))

        # Energy 
        print(format_header("Observations"))
        print(format_line(f"SoC: {float(self.info['soc_value'][-1]):.3f}, ==== SoC violation: {self.info['soc_violation'][-1]} ==== Clipped SoC: {self.info['soc_clipped'][-1]}")) 
        print(format_line(f"Load power: {self.p_l:.3f}"))
        print(format_line(f"Solar Power: {self.p_g:.3f}"))
        
        net_load = "Deficit" if self.p_l > self.p_g else "Surplus"
        print(format_line(f"Energy status: {net_load}"))


        print(format_header(f" Action space"))
        print(format_line(f"Action: {float(self.info["action_value"][-1]):.3f}, ==== Action violation: {self.info["action_violation"][-1]}, ==== clipped action: {self.info["action_clipped"][-1]}"))

        print(f"|{'-' * (box_width - 2)}|")

        print(format_header("Reward"))
        print(format_line(f"P_loss: {self.info["p_loss"][-1]}"))
        print(format_line(f"q_loss_step: {self.info["q_loss"][-1]}"))
        print(format_line(f"Q_loss_percent: {self.info["q_loss_percent"][-1]} % "))
        print(format_line(f"Accumulated Q_loss: {self.info["q_loss_cumulative"][-1]}"))
        print(format_line(f"Charge Throughput: {self.info["throughput_charge"][-1]}"))
        print(format_line(f"Reward: {self.info["reward"][-1]}"))

        print(f"|{'-' * (box_width - 2)}|")
        print(format_line(f"Health: {self.create_bar(self.info['q_loss_percent'][-1])}"))

        print(separator)



    def create_bar(self, percentage: float, width: int = 40) -> str:
            # Convert percentage to decimal (100% -> 1.0)
            decimal = percentage / 100.0
            # Ensure the value is between 0 and 1
            decimal = max(0, min(1, decimal))
            filled = int(decimal * width)
            return f"[{'█' * filled}{'-' * (width - filled)}] {percentage:.1f}%"










