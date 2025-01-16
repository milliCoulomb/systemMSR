# physics/thermohydraulics.py

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import logging
from methods.fvm import FVM
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

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