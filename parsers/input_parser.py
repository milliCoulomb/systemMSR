# parsers/input_parser.py

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pydantic import BaseModel, ValidationError, validator
import bisect
import numpy as np

class SchedulePointModel(BaseModel):
    time: float
    flow_rate: Optional[float] = None
    temperature: Optional[float] = None

class PumpModel(BaseModel):
    mode: str
    schedule: List[SchedulePointModel]

class SecondaryInletTempModel(BaseModel):
    schedule: List[SchedulePointModel]

class OperationalParametersModel(BaseModel):
    pump_primary: PumpModel
    pump_secondary: PumpModel
    secondary_inlet_temp: SecondaryInletTempModel

class MaterialsModel(BaseModel):
    primary_salt: dict
    secondary_salt: dict

class NuclearDataModel(BaseModel):
    diffusion_coefficient: float
    absorption_cross_section: float
    fission_cross_section: float
    nu_fission: float
    beta: float
    decay_constant: float
    kappa: float
    power: float

class GeometryModel(BaseModel):
    core_length: float
    exchanger_length: float
    core_radius: float
    cooling_loop_first_length: float
    cooling_loop_second_length: float
    secondary_loop_radius: float
    heat_exchanger_coefficient: float
    n_core: int
    n_exchanger: int
    n_cooling_loop_first_segment: int
    n_cooling_loop_second_segment: int

class SimulationModel(BaseModel):
    total_time: float
    time_step: float

class InputDeckModel(BaseModel):
    simulation: SimulationModel
    geometry: GeometryModel
    materials: MaterialsModel
    nuclear_data: NuclearDataModel
    operational_parameters: OperationalParametersModel

@dataclass
class InputDeck:
    simulation: SimulationModel
    geometry: GeometryModel
    materials: MaterialsModel
    nuclear_data: NuclearDataModel
    operational_parameters: OperationalParametersModel

    @staticmethod
    def from_yaml(file_path: str) -> 'InputDeck':
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        try:
            input_model = InputDeckModel(**data)
        except ValidationError as e:
            print("Input Deck Validation Error:")
            print(e.json())
            raise e
        
        return InputDeck(
            simulation=input_model.simulation,
            geometry=input_model.geometry,
            materials=input_model.materials,
            nuclear_data=input_model.nuclear_data,
            operational_parameters=input_model.operational_parameters
        )
    
    def get_pump_flow_rate(self, pump: PumpModel, current_time: float) -> float:
        """Interpolate the flow rate for a given pump at the current time."""
        times = np.array([point.time for point in pump.schedule])
        rates = np.array([point.flow_rate for point in pump.schedule])
        
        if current_time <= times[0]:
            return rates[0]
        elif current_time >= times[-1]:
            return rates[-1]
        else:
            idx = np.searchsorted(times, current_time) - 1
            t1, t2 = times[idx], times[idx + 1]
            r1, r2 = rates[idx], rates[idx + 1]
            # Linear interpolation
            return r1 + (r2 - r1) * (current_time - t1) / (t2 - t1)
    
    def get_secondary_inlet_temp(self, current_time: float) -> float:
        """Interpolate the secondary inlet temperature at the current time."""
        schedule = self.operational_parameters.secondary_inlet_temp.schedule
        times = np.array([point.time for point in schedule])
        temps = np.array([point.temperature for point in schedule])
        
        if current_time <= times[0]:
            return temps[0]
        elif current_time >= times[-1]:
            return temps[-1]
        else:
            idx = np.searchsorted(times, current_time) - 1
            t1, t2 = times[idx], times[idx + 1]
            temp1, temp2 = temps[idx], temps[idx + 1]
            # Linear interpolation
            return temp1 + (temp2 - temp1) * (current_time - t1) / (t2 - t1)