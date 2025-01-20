# physics/UnsteadyCoupler.py

from dataclasses import dataclass
from typing import List
import numpy as np
import logging
from utils.geometry import CoreGeometry, SecondaryLoopGeometry
from utils.states import ThermoHydraulicsState, NeutronicsState
from physics.thermo import ThermoHydraulicsSolver
from physics.neutronics import NeutronicsSolver
from utils.time_parameters import TimeParameters


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# define magic constants (residuals tolerance)
RESIDUAL_FLUX = 1e-4
RESIDUAL_TEMPERATURE = 1e-4


class UnsteadyCoupler:
    def __init__(
        self,
        th_solver: ThermoHydraulicsSolver,
        neutronics_solver: NeutronicsSolver,
        initial_neutronics_state: NeutronicsState,
        initial_th_state_primary: ThermoHydraulicsState,
        initial_th_state_secondary: ThermoHydraulicsState,
        operational_parameters: TimeParameters,
    ):
        self.th_solver = th_solver
        self.neutronics_solver = neutronics_solver
        self.initial_neutronics_state = initial_neutronics_state
        self.initial_th_state_primary = initial_th_state_primary
        self.initial_th_state_secondary = initial_th_state_secondary
        self.operational_parameters = operational_parameters

    def solve(
        self
    ):
        """
        Solve the unsteady-state coupled problem.
        """
        # given initial states, solve the coupled problem
        residual_flux = 1.0
        residual_temperature = 1.0
        # set the initial values of mass flow rates and temperature for the primary and secondary loops
        # step into the time loop
        current_th_state_primary = self.initial_th_state_primary
        current_th_state_secondary = self.initial_th_state_secondary
        current_neutronics_state = self.initial_neutronics_state
        list_th_primary = []
        list_th_secondary = []
        list_neutronics = []
        time_parameters = self.operational_parameters
        for i in range(time_parameters.num_time_steps):
            # at each time step there is a while loop to couple the neutronics and thermohydraulics
            # set the initial values of mass flow rates and temperature for the primary and secondary loops
            current_th_state_primary.flow_rate = time_parameters.pump_values_primary[i]
            current_th_state_secondary.flow_rate = time_parameters.pump_values_secondary[i]
            current_th_state_secondary.T_in = (
                time_parameters.secondary_inlet_temperature_values[i]
            )
            while (
                residual_flux > RESIDUAL_FLUX
                and residual_temperature > RESIDUAL_TEMPERATURE
            ):
                neutronic_step = self.neutronics_solver.make_neutronic_time_step(
                    th_state=current_th_state_primary,
                    neut_state=current_neutronics_state,
                    th_params=self.th_solver.params_primary,
                    dt=time_parameters.time_step,
                )
                # solve thermohydraulics
                th_primary_step, th_secondary_step = self.th_solver.make_time_step(
                    th_state_primary=current_th_state_primary,
                    th_state_secondary=current_th_state_secondary,
                    neutronic_state=neutronic_step,
                    dt=time_parameters.time_step,
                )
                # calculate residuals
                residual_flux = np.linalg.norm(
                    current_neutronics_state.phi - neutronic_step.phi
                ) / np.linalg.norm(current_neutronics_state.phi)
                residual_temperature = np.linalg.norm(
                    current_th_state_primary.temperature - th_primary_step.temperature
                ) / np.linalg.norm(current_th_state_primary.temperature)
                logger.info(
                    f"Residuals: flux={residual_flux}, temperature={residual_temperature}"
                )
                # update states
                current_th_state_primary = th_primary_step
                current_th_state_secondary = th_secondary_step
                current_neutronics_state = neutronic_step
            
            logger.info(f"Time step {i} completed.")
            list_th_primary.append(current_th_state_primary)
            list_th_secondary.append(current_th_state_secondary)
            list_neutronics.append(current_neutronics_state)
            residual_flux = 1.0
            residual_temperature = 1.0
            
        
        return list_th_primary, list_th_secondary, list_neutronics
