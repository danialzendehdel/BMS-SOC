import yaml
from dataclasses import dataclass

# --- Data Classes for Each Section ---

@dataclass
class DataPath:
    matlab_data: str
    matlab_data_test: str
    P_net: str

@dataclass
class BatteryConfig:
    nominal_capacity: float
    soc_init: float
    nominal_current: float
    nominal_Ah: float
    soc_min: float
    soc_max: float
    max_charge_rate_kW: float
    max_discharge_rate_kW: float
    Ns: float
    Np: float

@dataclass
class BatteryConfig_Tesla:
    model: str
    type: str
    nominal_capacity_kWh: float
    usable_capacity_kWh: float
    nominal_voltage_V: float
    max_charge_rate_kW: float
    max_discharge_rate_kW: float
    efficiency: float
    degradation_rate_per_step: float
    life_80_cyc: float

@dataclass
class RewardCoeff:
    coeff_q_loss: float
    coeff_p_loss: float

@dataclass
class EconomicConfig:
    constant_price: float

@dataclass
class EnvironmentConfig:
    battery: BatteryConfig
    reward_coeff: RewardCoeff
    economic: EconomicConfig

@dataclass
class TrainConfig:
    epsilon: float
    time_interval: float
    GA: bool

@dataclass
class Case:
    soc_range: str
    a: float
    b: float

    def __iter__(self):
        # This allows unpacking: a, b = instance_of_Case
        yield self.a
        yield self.b

@dataclass
class AgingModel:
    q_loss_eol: float
    constant_temperature: float
    Ea_J_per_mol: int
    Rg_J_per_molK: float
    h: float
    exponent_z: float
    case_l_45: Case
    case_b_45: Case

@dataclass
class FullConfig:
    data: DataPath
    environment: EnvironmentConfig
    train: TrainConfig
    aging_model: AgingModel
    batt_tesla: BatteryConfig_Tesla
# --- Loader Function ---

def load_full_config(filename: str) -> FullConfig:
    with open(filename, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Load data section.
    data_config = DataPath(**config_dict.get("data", {}))
    
    # Load environment section.
    env_dict = config_dict.get("environment", {})
    battery_config = BatteryConfig(**env_dict.get("battery", {}))
    
    reward_coeff_config = RewardCoeff(**env_dict.get("reward_coeff", {}))
    economic_config = EconomicConfig(**env_dict.get("economic", {}))
    environment_config = EnvironmentConfig(
        battery=battery_config,
        reward_coeff=reward_coeff_config,
        economic=economic_config
    )
    
    # Load train section.
    train_config = TrainConfig(**config_dict.get("train", {}))
    battery_config_tesla = BatteryConfig_Tesla(**config_dict.get("Tesla", {}))
    # Load aging_model section.
    aging_dict = config_dict.get("aging_model", {})
    case_l_45 = Case(**aging_dict.get("case_l_45", {}))
    case_b_45 = Case(**aging_dict.get("case_b_45", {}))
    aging_model_config = AgingModel(
        q_loss_eol=aging_dict["q_loss_eol"],
         constant_temperature=aging_dict["constant_temperature"],
         Ea_J_per_mol=aging_dict["Ea_J_per_mol"],
         Rg_J_per_molK=aging_dict["Rg_J_per_molK"],
         h=aging_dict["h"],
         exponent_z=aging_dict["exponent_z"],
         case_l_45=case_l_45,
         case_b_45=case_b_45
    )
    
    return FullConfig(
         data=data_config,
         environment=environment_config,
         train=train_config,
         aging_model=aging_model_config,
         batt_tesla=battery_config_tesla
    )


if __name__ == '__main__':
    config_file = 'Code/configuration/params_2.yml'
    config_instance = load_full_config(config_file)
    # print(config_instance)

    # Accessing some configuration values:
    # print("Nominal battery capacity:", config_instance.environment.battery.nominal_capacity)
    
    # # Unpack the case_l_45 a, b values using our custom iterator.
    # a, b = config_instance.aging_model.case_l_45
    # print("case_l_45 a:", a)
    # print("case_l_45 b:", b)
    
    # print("Constant temperature:", config_instance.aging_model.constant_temperature)

    # print(config_instance.train.time_interval)
    # print(config_instance.aging_model.q_loss_eol)
    # print(config_instance.environment.battery.max_charge_rate_kW)
    # print(config_instance.environment.battery.Ns)
    print(config_instance.batt_tesla)


