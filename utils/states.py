# utils/states.py

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import logging

@dataclass
class ThermoHydraulicsState:
    temperature: np.ndarray
    flow_rate: np.ndarray
    T_in: float  # Inlet temperature [K]

@dataclass
class NeutronicsState:
    phi: np.ndarray  # Neutron flux [n/m^2-s]
    C: np.ndarray  # Delayed neutron precursors concentration [n/m^3]
    keff: float  # Effective multiplication factor
    power: float  # Power [W]
    power_density: np.ndarray  # Power density [W/m^3]