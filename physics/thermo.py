# physics/thermohydraulics.py

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import logging
import methods.method as Method
from utils.geometry import CoreGeometry, SecondaryLoopGeometry


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThermoHydraulicsParameters:
    rho: float  # Density [kg/m^3]
    cp: float   # Specific heat [J/kg-K]
    k: float    # Thermal conductivity [W/m-K]
    heat_exchanger_coefficient: float  # [W/m^3-K]
    expansion_coefficient: float  # [1/K]

@dataclass
class ThermoHydraulicsState:
    core_temperature: np.ndarray  # Temperature distribution [K] in the core
    secondary_temperature: np.ndarray  # Temperature distribution [K] in the secondary loop
    flow_rate: float         # Mass flow rate [kg/s]

class ThermoHydraulicsSolver:
    def __init__(self, th_params: ThermoHydraulicsParameters, method: Method, core_geom: CoreGeometry, secondary_geom: SecondaryLoopGeometry):
        # Extract thermo-hydraulic parameters
        self.params = th_params
        self.core_geom = core_geom
        self.secondary_geom = secondary_geom
        self.method = method
        self.n_cells_primary = self.core_geom.n_cells_core + self.core_geom.n_cells_exchanger
        self.n_cells_secondary = self.secondary_geom.n_cells_first_loop + self.secondary_geom.n_cells_second_loop + self.secondary_geom.n_cells_exchanger