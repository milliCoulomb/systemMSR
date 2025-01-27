# physics/SteadyStateCoupler.py

from dataclasses import dataclass
from typing import List
import numpy as np
import logging
from utils.states import ThermoHydraulicsState, NeutronicsState
from physics.thermo import ThermoHydraulicsSolver
from physics.neutronics import NeutronicsSolver
import copy

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
        mode: str,
    ):
        self.th_solver = th_solver
        self.neutronics_solver = neutronics_solver
        # assess the mode of the simulation, can be "source_driven" or "criticality"
        self.mode = mode
        # check if the mode is valid
        if self.mode not in ["source_driven", "criticality"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Mode must be either 'source_driven' or 'criticality'."
            )

    def solve(
        self,
        th_state_primary: ThermoHydraulicsState,
        th_state_secondary: ThermoHydraulicsState,
        initial_neutronics_state: NeutronicsState,
        source: np.ndarray,
    ):
        """
        Solve the steady-state coupled problem.
        """
        # given initial states, solve the coupled problem
        residual_k = 1.0
        residual_flux = 1.0
        residual_temperature_steady = 1.0
        iteration = 0

        # Make deep copies of the states to avoid memory issues
        old_th_state_primary = th_state_primary
        old_th_state_secondary = th_state_secondary
        old_neutronics_state = initial_neutronics_state

        while (
            residual_k > RESIDUAL_KEFF
            or residual_flux > RESIDUAL_FLUX
            or residual_temperature > RESIDUAL_TEMPERATURE
        ):
            # Neutronics
            new_neutronics_state = self.neutronics_solver.solve_static(
                old_th_state_primary,
                self.th_solver.params_primary,
                source,
                override_mode=self.mode,
            )
            logger.debug(
                f"[Iteration {iteration}] keff={new_neutronics_state.keff}, power[150]={new_neutronics_state.power_density[150]}"
            )

            # Thermohydraulics
            new_th_primary_state, new_th_secondary_state = self.th_solver.solve_static(
                old_th_state_primary, old_th_state_secondary, new_neutronics_state
            )
            logger.debug(
                f"[Iteration {iteration}] New T(prim)[0..5] = {new_th_primary_state.temperature[:5]}"
            )
            logger.debug(
                f"[Iteration {iteration}] Old T(prim)[0..5] = {old_th_state_primary.temperature[:5]}"
            )

            # Residuals
            diff_p = new_th_primary_state.temperature - old_th_state_primary.temperature
            diff_s = (
                new_th_secondary_state.temperature - old_th_state_secondary.temperature
            )
            residual_temperature_1 = np.linalg.norm(diff_p) / np.linalg.norm(
                new_th_primary_state.temperature
            )
            residual_temperature_2 = np.linalg.norm(diff_s) / np.linalg.norm(
                new_th_secondary_state.temperature
            )
            residual_temperature = max(residual_temperature_1, residual_temperature_2)
            if self.mode == "criticality":
                residual_k = (
                    abs(new_neutronics_state.keff - old_neutronics_state.keff)
                    / new_neutronics_state.keff
                )
            else:
                residual_k = 0.0
            residual_flux = np.linalg.norm(
                new_neutronics_state.phi - old_neutronics_state.phi
            ) / np.linalg.norm(new_neutronics_state.phi)

            logger.debug(
                f"[Iteration {iteration}] Temperature diff (primary) first 5 = {diff_p[:5]}"
            )
            logger.info(
                f"Residuals: k={residual_k}, flux={residual_flux}, temperature={residual_temperature}"
            )

            # Overwrite old states
            old_th_state_primary = copy.deepcopy(new_th_primary_state)
            old_th_state_secondary = copy.deepcopy(new_th_secondary_state)
            old_neutronics_state = copy.deepcopy(new_neutronics_state)
        logger.debug(f"Converged in {iteration} iterations.")
        return old_th_state_primary, old_th_state_secondary, old_neutronics_state
