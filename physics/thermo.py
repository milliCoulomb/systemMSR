# physics/thermohydraulics.py

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import logging
import methods.method as Method
from utils.geometry import CoreGeometry, SecondaryLoopGeometry
from scipy.sparse import diags, spmatrix
from scipy.sparse import bmat
import scipy.sparse.linalg as spla
from utils.states import ThermoHydraulicsState, NeutronicsState


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MAGIC NUMBERS
THETA = 1.0


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
        method_primary: Method,
        core_geom: CoreGeometry,
        method_secondary: Method,
        secondary_geom: SecondaryLoopGeometry,
    ):
        # Extract thermo-hydraulic parameters
        self.params_primary = th_params_primary
        self.params_secondary = th_params_secondary
        self.core_geom = core_geom
        self.secondary_geom = secondary_geom
        self.method_primary = method_primary
        self.method_secondary = method_secondary
        self.n_cells_primary = (
            self.core_geom.n_cells_core + self.core_geom.n_cells_exchanger
        )
        self.n_cells_secondary = (
            self.secondary_geom.n_cells_first_loop
            + self.secondary_geom.n_cells_second_loop
            + self.secondary_geom.n_cells_exchanger
        )

    def build_temperature_advection_operator(
        self,
        th_state: ThermoHydraulicsState,
        periodic: bool = True,
        primary: bool = True,
    ):
        """
        Build the advection operator for the temperature field (mdot * cp * dT/dx)
        """
        # build the gradient operator
        if primary:
            grad = self.method_primary.build_grad(
                th_state.flow_rate
                * self.params_primary.cp
                * np.ones(self.n_cells_primary),
                periodic=periodic,
            )
        else:
            grad = self.method_secondary.build_grad(
                th_state.flow_rate
                * self.params_secondary.cp
                * np.ones(self.n_cells_secondary),
                periodic=periodic,
            )
        return grad

    def build_temperature_diffusion_operator(
        self,
        th_state: ThermoHydraulicsState,
        periodic: bool = True,
        primary: bool = True,
    ):
        """
        Build the diffusion operator for the temperature field (-k * A * d^2T/dx^2)
        """
        # build the stiffness operator
        if primary:
            stif = self.method_primary.build_stif(
                self.params_primary.k
                * np.ones(self.n_cells_primary)
                * self.core_geom.core_radius**2
                * np.pi,
                periodic=periodic,
            )
        else:
            stif = self.method_secondary.build_stif(
                self.params_secondary.k
                * np.ones(self.n_cells_secondary)
                * self.secondary_geom.loop_radius**2
                * np.pi,
                periodic=periodic,
            )
        return stif

    def build_heat_exchanger_operator(
        self,
        primary: bool = True,
    ):
        """
        Build the heat exchanger operator for the temperature field. (h * A)
        """
        if primary:
            heat_exchange = (
                np.concatenate(
                    [
                        np.zeros(self.core_geom.n_cells_core),
                        np.ones(self.core_geom.n_cells_exchanger),
                    ]
                )
                * self.params_primary.heat_exchanger_coefficient
                * (self.core_geom.core_radius**2 * np.pi)
            )
            matrix = self.method_primary.build_mass(heat_exchange)
        else:
            heat_exchange = (
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
            matrix = self.method_secondary.build_mass(heat_exchange)
        return matrix

    def assemble_matrix_static(
        self,
        th_state_primary: ThermoHydraulicsState,
        th_state_secondary: ThermoHydraulicsState,
    ):
        """
        Assemble the matrix for the static case
        """
        # build the operators
        diff_primary = self.build_temperature_diffusion_operator(
            th_state_primary, primary=True, periodic=True
        )
        adv_primary = self.build_temperature_advection_operator(
            th_state_primary, primary=True, periodic=True
        )
        heat_exchange_primary = self.build_heat_exchanger_operator(primary=True)
        diff_secondary = self.build_temperature_diffusion_operator(
            th_state_secondary, primary=False, periodic=False
        )
        adv_secondary = self.build_temperature_advection_operator(
            th_state_secondary, primary=False, periodic=False
        )
        heat_exchange_secondary = self.build_heat_exchanger_operator(primary=False)
        LHS_blocks = [
            [
                diff_primary + adv_primary + heat_exchange_primary,
                -heat_exchange_primary,
            ],
            [
                -heat_exchange_secondary,
                diff_secondary + adv_secondary + heat_exchange_secondary,
            ],
        ]
        LHS_mat = bmat(LHS_blocks)
        logger.debug("Matrix assembled for the static case.")
        return LHS_mat

    def assemble_matrix_time_dependent(
        self,
        th_state_primary: ThermoHydraulicsState,
        th_state_secondary: ThermoHydraulicsState,
    ):
        """
        Assemble the matrix for the time-dependent case.
        """
        # build the operators
        diff_primary = self.build_temperature_diffusion_operator(
            th_state_primary, primary=True, periodic=True
        ) / (
            self.params_primary.rho
            * self.params_primary.cp
            * self.core_geom.core_radius**2
            * np.pi
        )
        adv_primary = self.build_temperature_advection_operator(
            th_state_primary, primary=True, periodic=True
        ) / (
            self.params_primary.rho
            * self.params_primary.cp
            * self.core_geom.core_radius**2
            * np.pi
        )
        heat_exchange_primary = self.build_heat_exchanger_operator(primary=True) / (
            self.params_primary.rho
            * self.params_primary.cp
            * self.core_geom.core_radius**2
            * np.pi
        )
        diff_secondary = self.build_temperature_diffusion_operator(
            th_state_secondary, primary=False, periodic=False
        ) / (
            self.params_secondary.rho
            * self.params_secondary.cp
            * self.secondary_geom.loop_radius**2
            * np.pi
        )
        adv_secondary = self.build_temperature_advection_operator(
            th_state_secondary, primary=False, periodic=False
        ) / (
            self.params_secondary.rho
            * self.params_secondary.cp
            * self.secondary_geom.loop_radius**2
            * np.pi
        )
        heat_exchange_secondary = self.build_heat_exchanger_operator(primary=False) / (
            self.params_secondary.rho
            * self.params_secondary.cp
            * self.secondary_geom.loop_radius**2
            * np.pi
        )

        LHS_blocks = [
            [
                -diff_primary - adv_primary - heat_exchange_primary,
                heat_exchange_primary,
            ],
            [
                heat_exchange_secondary,
                -diff_secondary - adv_secondary - heat_exchange_secondary,
            ],
        ]
        LHS_mat = bmat(LHS_blocks)
        logger.debug("Matrix assembled for the time-dependent case.")
        return LHS_mat

    def build_steady_state_rhs_vector(
        self,
        th_state_primary: ThermoHydraulicsState,
        th_state_secondary: ThermoHydraulicsState,
        neutronic_state: NeutronicsState,
    ):
        """
        Build the right-hand side vector for the steady-state problem (Pv * A + )
        """
        # we need to impose an inhomogeneous boundary condition for the secondary loop (T(0) = T_in)
        # therefore we need to modify the RHS vector so that the source term exactly cancels the term in the matrix
        # CAREFUL THIS IS DISCRETIZATION DEPENDENT
        boundary_condition = th_state_secondary.T_in * (
            th_state_secondary.flow_rate
            * self.params_secondary.cp
            / self.secondary_geom.dx[0]
            + self.params_secondary.k
            / self.secondary_geom.dx[0] ** 2
            * np.pi
            * self.secondary_geom.loop_radius**2
        )
        source_term_secondary = np.concatenate(
            [
                np.array([boundary_condition]),
                np.zeros(self.secondary_geom.n_cells_first_loop - 1),
                np.zeros(self.secondary_geom.n_cells_exchanger),
                np.zeros(self.secondary_geom.n_cells_second_loop),
            ]
        )
        # CAREFUL POWER DENSITY IS ALREADY MULTIPLIED BY THE AREA
        power_source = neutronic_state.power_density
        RHS_vector = np.concatenate([power_source, source_term_secondary])
        return RHS_vector

    def build_time_dependent_rhs_vector(
        self,
        th_state_primary: ThermoHydraulicsState,
        th_state_secondary: ThermoHydraulicsState,
        neutronic_state: NeutronicsState,
    ):
        """
        Build the right-hand side vector for the time-dependent problem
        """
        boundary_condition = th_state_secondary.T_in * (
            th_state_secondary.flow_rate
            * self.params_secondary.cp
            / self.secondary_geom.dx[0]
            + self.params_secondary.k
            / self.secondary_geom.dx[0] ** 2
            * np.pi
            * self.secondary_geom.loop_radius**2
        )
        source_term_secondary = np.concatenate(
            [
                np.array([boundary_condition]),
                np.zeros(self.secondary_geom.n_cells_first_loop - 1),
                np.zeros(self.secondary_geom.n_cells_exchanger),
                np.zeros(self.secondary_geom.n_cells_second_loop),
            ]
        ) / (
            self.params_secondary.rho
            * self.params_secondary.cp
            * self.secondary_geom.loop_radius**2
            * np.pi
        )
        # CAREFUL POWER DENSITY IS ALREADY MULTIPLIED BY THE AREA
        power_source = neutronic_state.power_density / (
            self.params_primary.rho
            * self.params_primary.cp
            * self.core_geom.core_radius**2
            * np.pi
        )
        RHS_vector = np.concatenate([power_source, source_term_secondary])
        return RHS_vector

    def solve_static(
        self,
        th_state_primary: ThermoHydraulicsState,
        th_state_secondary: ThermoHydraulicsState,
        neutronic_state: NeutronicsState,
    ):
        """
        Solve the static thermo-hydraulic problem
        """
        LHS = self.assemble_matrix_static(th_state_primary, th_state_secondary)
        rhs_vector = self.build_steady_state_rhs_vector(
            th_state_primary, th_state_secondary, neutronic_state
        )
        T = spla.spsolve(LHS, rhs_vector)
        logger.debug("Static thermo-hydraulic problem solved.")
        th_state_primary.temperature = T[: self.n_cells_primary]
        th_state_secondary.temperature = T[self.n_cells_primary :]
        return th_state_primary, th_state_secondary

    def make_time_step(
        self,
        th_state_primary: ThermoHydraulicsState,
        th_state_secondary: ThermoHydraulicsState,
        neutronic_state: NeutronicsState,
        dt: float,
    ):
        """
        Make a time step for the transient problem
        """
        operator = self.assemble_matrix_time_dependent(
            th_state_primary, th_state_secondary
        )
        rhs_vector = self.build_time_dependent_rhs_vector(
            th_state_primary, th_state_secondary, neutronic_state
        )
        identity = diags(
            [np.ones(self.n_cells_primary + self.n_cells_secondary)],
            [0],
            shape=(
                self.n_cells_primary + self.n_cells_secondary,
                self.n_cells_primary + self.n_cells_secondary,
            ),
        )
        LHS = identity - THETA * dt * operator
        RHS = identity + (1 - THETA) * dt * operator
        T = spla.spsolve(
            LHS,
            RHS
            @ np.concatenate(
                [th_state_primary.temperature, th_state_secondary.temperature]
            )
            + dt * rhs_vector,
        )
        T_primary = T[: self.n_cells_primary]
        T_secondary = T[self.n_cells_primary :]
        # iniatiate new state
        th_state_primary_new = ThermoHydraulicsState(
            flow_rate=th_state_primary.flow_rate,
            temperature=T_primary,
            T_in=th_state_primary.T_in,
        )
        th_state_secondary_new = ThermoHydraulicsState(
            flow_rate=th_state_secondary.flow_rate,
            temperature=T_secondary,
            T_in=th_state_secondary.T_in,
        )
        logger.debug("Transient thermo-hydraulic problem solved.")
        return th_state_primary_new, th_state_secondary_new
