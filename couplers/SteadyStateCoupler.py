# physics/neutronics.py

from dataclasses import dataclass
from typing import List
import numpy as np
import logging
from utils.geometry import CoreGeometry, SecondaryLoopGeometry
from utils.states import ThermoHydraulicsState, NeutronicsState
from physics.thermo import ThermoHydraulicsSolver
from physics.neutronics import NeutronicsSolver


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# define magic constants (residuals tolerance)
RESIDUAL_KEFF = 1e-6
RESIDUAL_FLUX = 1e-5
RESIDUAL_TEMPERATURE = 1e-5


class SteadyStateCoupler:
    def __init__(
        self,
        th_solver: ThermoHydraulicsSolver,
        neutronics_solver: NeutronicsSolver,
    ):
        self.th_solver = th_solver
        self.neutronics_solver = neutronics_solver

    def solve(
        self,
        th_state_primary: ThermoHydraulicsState,
        th_state_secondary: ThermoHydraulicsState,
        initial_neutronics_state: NeutronicsState,
    ):
        """
        Solve the steady-state coupled problem.
        """
        # given initial states, solve the coupled problem
        residual_k = 1.0
        residual_flux = 1.0
        residual_temperature = 1.0
        while (
            residual_k > RESIDUAL_KEFF
            or residual_flux > RESIDUAL_FLUX
            or residual_temperature > RESIDUAL_TEMPERATURE
        ):
            # solve neutronics first
            neutronic_step = self.neutronics_solver.solve_static(
                th_state_primary, self.th_solver.params_primary
            )
            # solve thermohydraulics
            th_primary_step, th_secondary_step = self.th_solver.solve_static(
                th_state_primary=th_state_primary,
                th_state_secondary=th_state_secondary,
                neutronic_state=neutronic_step,
            )
            # calculate residuals
            residual_k = (
                np.abs(neutronic_step.keff - initial_neutronics_state.keff)
                / initial_neutronics_state.keff
            )
            residual_flux = np.linalg.norm(
                neutronic_step.phi - initial_neutronics_state.phi
            ) / np.linalg.norm(initial_neutronics_state.phi)
            residual_temperature = np.linalg.norm(
                th_primary_step.temperature - th_state_primary.temperature
            ) / np.linalg.norm(th_state_primary.temperature)
            # update states
            th_state_primary = th_primary_step
            th_state_secondary = th_secondary_step
            initial_neutronics_state = neutronic_step
            logger.info(
                f"Residuals: k={residual_k}, flux={residual_flux}, temperature={residual_temperature}"
            )
        return th_state_primary, th_state_secondary, initial_neutronics_state
