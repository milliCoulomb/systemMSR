# physics/neutronics.py

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from parsers.input_parser import InputDeck
import logging
import methods.method as Method
import utils.geometry as Geometry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NeutronicsParameters:
    D: float  # Diffusion coefficient [m]
    Sigma_a: float  # Absorption cross-section [1/m]
    Sigma_f: float  # Fission cross-section [1/m]
    nu_fission: float  # Average neutrons per fission
    beta: float  # Delayed neutron fraction
    Lambda: float  # Decay constant [1/s]

@dataclass
class NeutronicsState:
    phi: np.ndarray  # Neutron flux [n/cm^2-s]
    C: np.ndarray    # Delayed neutron precursors concentration [n/cm^3]

class NeutronicsSolver:
    def __init__(self, input_deck: InputDeck, method: Method, geometry: Geometry):
        # Extract nuclear data
        nuc = input_deck.nuclear_data
        self.params = NeutronicsParameters(
            D=nuc.diffusion_coefficient,
            Sigma_a=nuc.absorption_cross_section,
            Sigma_f=nuc.fission_cross_section,
            nu_fission=nuc.nu_fission,
            beta=nuc.beta,
            Lambda=nuc.decay_constant
        )
        
        # Extract geometry
        geom = geometry
        self.n_cells = geom.n_cells_core + geom.n_cells_exchanger
        self.length = geom.core_length + geom.exchanger_length
        self.dx_core = geom.core_length / geom.n_core
        self.dx_exchanger = geom.exchanger_length / geom.n_exchanger
        self.dx = np.array([self.dx_core] * geom.n_core + [self.dx_exchanger] * geom.n_exchanger)
        
        # Initialize state
        self.phi = np.ones(self.n_cells)  # Initial neutron flux [n/m^2-s]
        self.C = np.zeros(self.n_cells)   # Initial precursor concentrations [n/m^3]

        self.method = method
        
        # Time step will be handled externally
        logger.info(f"Neutronics Solver initialized with {self.n_cells} cells.")
    
    def update_nuclear_parameters(self, temperature: np.ndarray) -> np.ndarray:
        """
        Update temperature-dependent cross sections and returns arrays of Sigma_a, Sigma_f, and D.
        """
        # load density and expansion coefficients from input deck
        rho = self.input_deck.materials.primary_salt['density']
        alpha = self.input_deck.materials.primary_salt['expansion']
        T_ref = 922.0  # Reference temperature [K]
        # calculate the temperature-dependent cross sections
        Sigma_a = self.params.Sigma_a * (1 - alpha * (temperature - T_ref))
        Sigma_f = self.params.Sigma_f * (1 - alpha * (temperature - T_ref))
        D = self.params.D / (1 + alpha * (temperature - T_ref))
        logger.debug("Nuclear parameters updated based on temperature.")
        return Sigma_a, Sigma_f, D
    
    # define a method to assemble the eigenproblem matrices for the vector (phi, C) at a given time step and temperature profile
    def assemble_matrix_static(self, time: float, temperature: np.ndarray) -> np.ndarray:
        """
        Assemble the matrix for the neutron flux and precursor concentration eigenvalue problem at a given time step and temperature profile.
        """
        # evaluate the cross sections and diffusion coefficient at the given temperature
        Sigma_a, Sigma_f, D = self.update_nuclear_parameters(temperature)
        # build the FVM matrices
        # diffusion matrix for the neutron flux
        diffusion_flux = self.method.build_stif(D)
        # absorption matrix for the neutron flux
        absorption_flux = self.method.build_mass(Sigma_a)
        # fission matrix for the neutron flux
        fission_flux = self.method.build_mass(Sigma_f) * self.params.nu_fission * (1 - self.params.beta)
        # delayed neutron precursor matrix (advection term)
        precursor_advection = self.method.build_grad(self.params.Lambda)


        
        logger.debug("Matrix assembled for the eigenvalue problem.")
        return A
