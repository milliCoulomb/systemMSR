# physics/neutronics.py

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.sparse import diags
from scipy.sparse import bmat
import scipy.sparse.linalg as spla
import logging
import methods.method as Method
from utils.geometry import CoreGeometry
from utils.states import ThermoHydraulicsState, NeutronicsState
from physics.thermo import ThermoHydraulicsParameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MAGIC CONSTANTS
THETA = 1.0  # Implicitness parameter


@dataclass
class NeutronicsParameters:
    D: float  # Diffusion coefficient [m]
    Sigma_a: float  # Absorption cross-section [1/m]
    Sigma_f: float  # Fission cross-section [1/m]
    nu_fission: float  # Average neutrons per fission
    beta: float  # Delayed neutron fraction
    Lambda: float  # Decay constant [1/s]
    kappa: float  # energy release per fission [J]
    power: float  # Power [W]
    neutron_velocity: float  # Neutron velocity [m/s]


class NeutronicsSolver:
    def __init__(
        self,
        neutronic_parameters: NeutronicsParameters,
        method: Method,
        geometry: CoreGeometry,
    ):
        # Extract nuclear data
        self.params = neutronic_parameters

        # Extract geometry
        self.geom = geometry
        self.n_cells = self.geom.n_cells_core + self.geom.n_cells_exchanger
        self.dx = self.geom.dx

        # Initialize state
        self.phi = np.ones(self.n_cells)  # Initial neutron flux [n/m^2-s]
        self.C = np.zeros(self.n_cells)  # Initial precursor concentrations [n/m^3]

        self.method = method

        # Time step will be handled externally
        logger.info(f"Neutronics Solver initialized with {self.n_cells} cells.")

    def update_nuclear_parameters(
        self, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters
    ) -> np.ndarray:
        """
        Update temperature-dependent cross sections and returns arrays of Sigma_a, Sigma_f, and D.
        """
        # load density and expansion coefficients from input deck
        alpha = th_params.expansion_coefficient
        T_ref = 922.0 * np.ones_like(self.dx)
        if len(th_state.temperature) != len(self.dx):
            raise ValueError("Temperature profile and geometry mismatch.")
        # calculate the temperature-dependent cross sections
        Sigma_a = self.params.Sigma_a * (1 - alpha * (th_state.temperature - T_ref))
        Sigma_f = self.params.Sigma_f * (1 - alpha * (th_state.temperature - T_ref))
        D = self.params.D * (1 + alpha * (th_state.temperature - T_ref))
        logger.debug("Nuclear parameters updated based on temperature.")
        return Sigma_a, Sigma_f, D

    def build_removal_op_flux(
        self, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters
    ) -> np.ndarray:
        """
        Build the removal operator for the neutron flux. (diffusion + absorption, or -d_x(D d_x(phi)) + Sigma_a phi)
        """
        # evaluate the cross sections and diffusion coefficient at the given temperature
        Sigma_a, _, D = self.update_nuclear_parameters(
            th_state=th_state, th_params=th_params
        )
        diffusion_flux = self.method.build_stif(D)
        absorption_flux = self.method.build_mass(Sigma_a)
        logger.debug("Removal operator built for the neutron flux.")
        return diffusion_flux + absorption_flux

    def build_removal_op_precursor(
        self, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters
    ) -> np.ndarray:
        """
        Build the removal operator for the delayed neutron precursors. (advection + decay or v d_x(C)) + Lambda C)
        """
        # evaluate the cross sections and diffusion coefficient at the given temperature
        decay_precursor = self.method.build_mass(self.params.Lambda)
        velocity = (
            th_state.flow_rate
            / (th_params.rho * np.pi * self.geom.core_radius**2)
            * np.ones_like(self.dx)
        )
        precursor_advection = self.method.build_grad(velocity)
        logger.debug("Removal operator built for the delayed neutron precursors.")
        return precursor_advection + decay_precursor

    def build_precursor_production_op(
        self, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters
    ) -> np.ndarray:
        """
        Build the operator for the production of delayed neutron precursors (nu * Sigma_f * phi * beta).
        """
        # evaluate the cross sections and diffusion coefficient at the given temperature
        _, Sigma_f, _ = self.update_nuclear_parameters(
            th_state=th_state, th_params=th_params
        )
        Sigma_f = np.concatenate(
            [Sigma_f[: self.geom.n_cells_core], np.zeros(self.geom.n_cells_exchanger)]
        )
        precursor_production = (
            self.method.build_mass(Sigma_f) * self.params.nu_fission * self.params.beta
        )
        logger.debug("Operator built for the production of delayed neutron precursors.")
        return precursor_production

    def build_precursor_decay_op(self) -> np.ndarray:
        """
        Build the operator for the decay of delayed neutron precursors (Lambda).
        """
        decay_precursor = self.method.build_mass(self.params.Lambda)
        logger.debug("Operator built for the decay of delayed neutron precursors.")
        return decay_precursor

    def build_prompt_fission_op(
        self, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters
    ) -> np.ndarray:
        """
        Build the operator for the prompt fission neutrons (nu * Sigma_f * phi * (1 - beta)).
        """
        # evaluate the cross sections and diffusion coefficient at the given temperature
        _, Sigma_f, _ = self.update_nuclear_parameters(
            th_state=th_state, th_params=th_params
        )
        Sigma_f = np.concatenate(
            [Sigma_f[: self.geom.n_cells_core], np.zeros(self.geom.n_cells_exchanger)]
        )
        fission_flux = (
            self.method.build_mass(Sigma_f)
            * self.params.nu_fission
            * (1 - self.params.beta)
        )
        logger.debug("Operator built for the prompt fission neutrons.")
        return fission_flux

    def build_empty_op(self) -> np.ndarray:
        """
        Build an empty operator.
        """
        empty = diags(np.zeros_like(self.dx), 0, shape=(len(self.dx), len(self.dx)))
        logger.debug("Empty operator built.")
        return empty

    def assemble_matrix_static(
        self, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters
    ) -> np.ndarray:
        """
        Assemble the matrix for the neutron flux and precursor concentration eigenvalue problem at a given time step and temperature profile.
        """
        # build the operators
        removal_flux = self.build_removal_op_flux(th_state, th_params)
        removal_precs = self.build_removal_op_precursor(th_state, th_params)
        decay_precursor = self.build_precursor_decay_op()
        precursor_production = self.build_precursor_production_op(th_state, th_params)
        fission_flux = self.build_prompt_fission_op(th_state, th_params)
        empty = self.build_empty_op()
        # assemble the matrices
        LHS_blocks = [
            [removal_flux, -decay_precursor],
            [empty, removal_precs],
        ]
        RHS_blocks = [
            [fission_flux, empty],
            [precursor_production, empty],
        ]
        print(f"Shape of empty: {empty.shape}")
        LHS_mat = bmat(LHS_blocks)
        RHS_mat = bmat(RHS_blocks)
        logger.debug("Matrix assembled for the eigenvalue problem.")
        return LHS_mat, RHS_mat
    
    def assemble_matrix_time_dependent(
        self, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters
    ) -> np.ndarray:
        """
        Assemble the matrix for the neutron flux and precursor concentration time-dependent problem at a given time step and temperature profile.
        """
        # build the operators
        removal_flux = self.build_removal_op_flux(th_state, th_params) * self.params.neutron_velocity # -d_x(D d_x(phi)) + Sigma_a phi
        removal_precs = self.build_removal_op_precursor(th_state, th_params) # v d_x(C) + Lambda C
        decay_precursor = self.build_precursor_decay_op() * self.params.neutron_velocity # Lambda * v
        precursor_production = self.build_precursor_production_op(th_state, th_params) # nu * Sigma_f * phi * beta
        fission_flux = self.build_prompt_fission_op(th_state, th_params) * self.params.neutron_velocity # nu * Sigma_f * phi * (1 - beta) * v
        empty = self.build_empty_op()
        # assemble the matrices that apply to the vector (phi, C)
        operator = [[-removal_flux + fission_flux, decay_precursor], [precursor_production, -removal_precs]]
        return bmat(operator)

    def flux_normalization(
        self,
        neut_state: NeutronicsState,
        th_state: ThermoHydraulicsState,
        th_params: ThermoHydraulicsParameters,
    ) -> NeutronicsState:
        """
        Normalize the neutron flux distribution to the power.
        """
        # calculate the power
        _, sigma_f, _ = self.update_nuclear_parameters(th_state, th_params)
        mask = np.concatenate(
            [np.ones(self.geom.n_cells_core), np.zeros(self.geom.n_cells_exchanger)]
        )
        sigma_f = sigma_f * mask
        power = (
            np.sum(neut_state.phi * sigma_f * self.dx)
            * np.pi
            * self.geom.core_radius**2
            * self.params.kappa
        )
        # normalize the flux
        phi = neut_state.phi * self.params.power / power
        C = neut_state.C * self.params.power / power
        power_density = (
            phi * sigma_f * self.params.kappa * np.pi * self.geom.core_radius**2
        )
        logger.debug("Neutron flux normalized.")
        return NeutronicsState(
            phi=phi, C=C, keff=neut_state.keff, power=power, power_density=power_density
        )
    
    def compute_power(self, neut_state: NeutronicsState, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters) -> float:
        """
        Compute the power from the neutron flux distribution.
        """
        _, sigma_f, _ = self.update_nuclear_parameters(th_state, th_params)
        mask = np.concatenate(
            [np.ones(self.geom.n_cells_core), np.zeros(self.geom.n_cells_exchanger)]
        )
        sigma_f = sigma_f * mask
        power = (
            np.sum(neut_state.phi * sigma_f * self.dx)
            * np.pi
            * self.geom.core_radius**2
            * self.params.kappa
        )
        return power

    def solve_static(
        self, th_state: ThermoHydraulicsState, th_params: ThermoHydraulicsParameters
    ) -> NeutronicsState:
        """
        Solve the neutron flux and precursor concentration eigenvalue problem at a given time step and temperature profile.
        """
        # assemble the matrices
        LHS_mat, RHS_mat = self.assemble_matrix_static(th_state, th_params)
        # solve the eigenvalue problem
        eigvals, eigvecs = spla.eigs(
            RHS_mat,
            k=1,
            M=LHS_mat,
        )
        # extract the solution
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # fundamental mode is the largest eigenvalue, so the last
        phi = np.real(eigvecs[: self.n_cells, -1])
        C = np.real(eigvecs[self.n_cells :, -1])
        logger.debug("Eigenvalue problem solved.")
        return self.flux_normalization(
            NeutronicsState(
                phi=phi,
                C=C,
                keff=np.real(eigvals[-1]),
                power=self.params.power,
                power_density=np.zeros_like(phi),
            ),
            th_state,
            th_params,
        )
    
    def make_neutronic_time_step(
        self,
        th_state: ThermoHydraulicsState,
        th_params: ThermoHydraulicsParameters,
        neut_state: NeutronicsState,
        dt: float,
    ) -> NeutronicsState:
        """
        Make a time step for the neutron flux and precursor concentration.
        """
        operator = self.assemble_matrix_time_dependent(th_state, th_params)
        identity = diags(np.ones_like(self.dx), 0, shape=(len(self.dx), len(self.dx)))
        LHS = identity - THETA * dt * operator
        RHS = identity + (1 - THETA) * dt * operator
        # solve the system
        phi_c_new = spla.spsolve(LHS, RHS @ np.concatenate([neut_state.phi, neut_state.C]))
        # extract the solution and put it in the state
        phi_new = phi_c_new[: self.n_cells]
        C_new = phi_c_new[self.n_cells :]
        # compute the new power lineic density
        _, sigma_f, _ = self.update_nuclear_parameters(th_state, th_params)
        pv = sigma_f * self.params.kappa * np.pi * self.geom.core_radius**2 * phi_new
        new_power = np.sum(pv * self.dx)
        # initiate a new state
        new_neut_state = NeutronicsState(
            phi=phi_new,
            C=C_new,
            keff=neut_state.keff,
            power=new_power,
            power_density=pv,
        )
        logger.debug("Neutronic time step made.")
        return new_neut_state


        
