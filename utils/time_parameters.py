# utils/time_parameters.py
# utility class that contains the operational parameters, the number of time steps and the values at each time step using linear interpolation

from dataclasses import dataclass
from typing import List
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

@dataclass
class TimeParameters:
    """
    Class that contains the operational parameters, the number of time steps and the values at each time step using linear interpolation.
    """

    # operational parameters
    time_step: float
    total_time: float
    num_time_steps: int

    # operational values
    pump_values_primary: np.ndarray
    time_values_primary_pump: np.ndarray

    pump_values_secondary: np.ndarray
    time_values_secondary_pump: np.ndarray

    secondary_inlet_temperature_values: np.ndarray
    time_values_secondary_inlet_temperature: np.ndarray

    # now we use the class to define the parameters at each time step

    def __post_init__(self):
        """
        Initialize the time parameters.
        """
        self.time_values = np.linspace(0, self.total_time, self.num_time_steps)

        # interpolate the pump values
        self.pump_values_primary = np.interp(
            self.time_values, self.time_values_primary_pump, self.pump_values_primary
        )

        self.pump_values_secondary = np.interp(
            self.time_values, self.time_values_secondary_pump, self.pump_values_secondary
        )

        # interpolate the secondary inlet temperature values
        self.secondary_inlet_temperature_values = np.interp(
            self.time_values,
            self.time_values_secondary_inlet_temperature,
            self.secondary_inlet_temperature_values,
        )

        logger.info("Time parameters initialized.")