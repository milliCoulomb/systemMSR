# physics/thermohydraulics.py

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import logging
import methods.method as Method
from utils.geometry import CoreGeometry, SecondaryLoopGeometry
from scipy.sparse import diags
from scipy.sparse import bmat
import scipy.sparse.linalg as spla
from utils.states import ThermoHydraulicsState, NeutronicsState


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ThermoHydraulicsParameters:
    rho: float  # Density [kg/m^3]
    cp: float  # Specific heat [J/kg-K]
    k: float  # Thermal conductivity [W/m-K]
    heat_exchanger_coefficient: float  # [W/m^3-K]
    expansion_coefficient: float  # [1/K]

class ThermoHydraulicsSolver:
    def __init__(
        self,
        th_params_primary: ThermoHydraulicsParameters,
        th_params_secondary: ThermoHydraulicsParameters,
        method: Method,
        core_geom: CoreGeometry,
        secondary_geom: SecondaryLoopGeometry,
    ):
        # Extract thermo-hydraulic parameters
        self.params_primary = th_params_primary
        self.params_secondary = th_params_secondary
        self.core_geom = core_geom
        self.secondary_geom = secondary_geom
        self.method = method
        self.n_cells_primary = (
            self.core_geom.n_cells_core + self.core_geom.n_cells_exchanger
        )
        self.n_cells_secondary = (
            self.secondary_geom.n_cells_first_loop
            + self.secondary_geom.n_cells_second_loop
            + self.secondary_geom.n_cells_exchanger
        )

    def assemble_matrix_static(self, th_state_primary: ThermoHydraulicsState, th_state_secondary: ThermoHydraulicsState):
        """
        Assemble the matrix for the static case
        """
        # advection for the primary loop
        adv_primary = self.method.build_grad(
            th_state_primary.flow_rate * self.params_primary.cp * np.ones(self.n_cells_primary)
        )
        # diffusion for the primary loop, careful to multiply by the area
        diff_primary = self.method.build_stif(
            self.params_primary.k
            * np.ones(self.n_cells_primary)
            * self.core_geom.core_radius**2
            * np.pi
        )
        # heat exchanger term, mask because it only applies to the exchanger in the primary loop
        heat_exchange_primary = (
            np.concatenate(
                [
                    np.zeros(self.core_geom.n_cells_core),
                    np.ones(self.core_geom.n_cells_exchanger),
                ]
            )
            * self.params_primary.heat_exchanger_coefficient
            * (self.core_geom.core_radius**2 * np.pi)
        )
        # advection for the secondary loop
        adv_secondary = self.method.build_grad(
            th_state_secondary.flow_rate * self.params_secondary.cp * np.ones(self.n_cells_secondary)
        )
        # diffusion for the secondary loop
        diff_secondary = self.method.build_stif(
            self.params_secondary.k
            * np.ones(self.n_cells_secondary)
            * self.secondary_geom.loop_radius**2
            * np.pi
        )
        # heat exchanger term, mask because it only applies to the exchanger in the secondary loop, so after n_cells_first_loop and before n_cells_first_loop + n_cells_exchanger
        heat_exchange_secondary = (
            np.concatenate(
                [
                    np.zeros(self.secondary_geom.n_cells_first_loop),
                    np.ones(self.secondary_geom.n_cells_exchanger),
                    np.zeros(self.secondary_geom.n_cells_second_loop),
                ]
            )
            * self.params_secondary.heat_exchanger_coefficient
            * (self.secondary_geom.loop_radius**2 * np.pi)
        )
        # assemble the matrix
        LHS_blocks = [
            [diff_primary + adv_primary + heat_exchange_primary, -heat_exchange_primary],
            [-heat_exchange_secondary, diff_secondary + adv_secondary + heat_exchange_secondary],
        ]
        LHS_mat = bmat(LHS_blocks)
        logger.debug("Matrix assembled for the static case.")
        return LHS_mat
    
    def solve_static(self, th_state_primary: ThermoHydraulicsState, th_state_secondary: ThermoHydraulicsState, neutronic_state: NeutronicsState):
        """
        Solve the static thermo-hydraulic problem
        """
        # assemble the matrix
        LHS = self.assemble_matrix_static(th_state_primary, th_state_secondary)
        # solve the linear system
        T = spla.spsolve(LHS, np.zeros_like(th_state_primary.core_temperature))
        logger.debug("Static thermo-hydraulic problem solved.")
        return T


