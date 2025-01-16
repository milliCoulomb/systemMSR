# utils/states.py

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import logging

@dataclass
class ThermoHydraulicsState:
    temperature: np.ndarray
    flow_rate: float  # Mass flow rate [kg/s]

@dataclass
class NeutronicsState:
    phi: np.ndarray  # Neutron flux [n/cm^2-s]
    C: np.ndarray  # Delayed neutron precursors concentration [n/cm^3]
    keff: float  # Effective multiplication factor
    power: float  # Power [W]