import yaml 
from dataclasses import dataclass

@dataclass
class Case:
    soc_range: str
    a: float
    b: float

    def __iter__(self):
        # Yield only a and b for unpacking
        yield self.a
        yield self.b

@dataclass
class Eqn4:
    Ea_J_per_mol: int
    Rg_J_per_molK: float
    h: float
    exponent_z: float
    case_b_45: Case
    case_l_45: Case
    constant_temperature: float

@dataclass
class AgingModel:
    eqn_4: Eqn4

def load_env_config(filename: str) -> AgingModel:
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)

    aging_model_data = config['aging_model']
    eqn4_data = aging_model_data['eqn_4']

    # Create Case instances from the nested dictionaries.
    case_1 = Case(**eqn4_data['case_b_45'])
    case_2 = Case(**eqn4_data['case_l_45'])
    
    # Create an Eqn4 instance using the data from the YAML.
    eqn_4 = Eqn4(
        Ea_J_per_mol=eqn4_data['Ea_J_per_mol'],
        Rg_J_per_molK=eqn4_data['Rg_J_per_molK'],
        h=eqn4_data['h'],
        exponent_z=eqn4_data['exponent_z'],
        case_b_45=case_1,
        case_l_45=case_2,
        constant_temperature=eqn4_data["constant_temperature"]
    )
    
    return AgingModel(eqn_4=eqn_4)



if __name__ == '__main__':
    config_file = 'configuration/params.yml'
    aging_model_config = load_env_config(config_file)
    # print(aging_model_config)

    print(aging_model_config.eqn_4.case_b_45)

    a, b = aging_model_config.eqn_4.case_l_45
    print(a)
    print(b)
    print(aging_model_config.eqn_4.constant_temperature)