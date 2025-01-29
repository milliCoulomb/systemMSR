# physics/turbulence.py
# given a mass flow rate vector on a mesh at a with a time step dt, it returns the mass flow rate vector at the next time step
# turbulence process is modeled as an Ornstein-Uhlenbeck process. First we need to obtain the Reynolds number.
# Then we calculate the turbulence intensity as 0.16 * Re^(-1/8) and the turbulence length scale as 0.07 * L

import numpy as np
from scipy.ndimage import gaussian_filter1d


def calculate_typical_length(core_radius, core_length) -> float:
    """
    Calculate the typical length of the core.
    """
    core_volume = np.pi * core_radius**2 * core_length
    core_surface = 2 * np.pi * core_radius * core_length
    return core_volume / core_surface


def Re(flow_rate, core_radius, core_length, density, viscosity) -> float:
    """
    Calculate the Reynolds number given the nominal flow rate, core radius, core length, density, and viscosity.
    """
    typical_length = calculate_typical_length(core_radius, core_length)
    return (flow_rate[0] / (np.pi * core_radius**2)) * typical_length / (viscosity)

def turbulence_intensity(Re) -> float:
    """
    Calculate the turbulence intensity given the Reynolds number.
    """
    return 0.16 * Re**(-1/8)

def turbulence_length_scale(core_radius, core_length) -> float:
    """
    Calculate the turbulence length scale given the core radius and core length.
    """
    typical_length = calculate_typical_length(core_radius, core_length)
    return 0.07 * typical_length

def turbulence_process(flow_rate, old_perturbuation, core_radius, core_length, density, viscosity, dt, dx) -> np.ndarray:
    """
    Given a mass flow rate vector on a mesh at a with a time step dt, it returns the mass flow rate vector at the next time step.
    """
    Re_number = Re(flow_rate, core_radius, core_length, density, viscosity)
    turbulence_int = turbulence_intensity(Re_number)
    # compute the characteristic time scale (ratio of the velocity scale to the length scale)
    turbulence_length = turbulence_length_scale(core_radius, core_length)
    # infer velocity from flow rate
    velocity = flow_rate[0] / (np.pi * core_radius**2 * density)
    # time scale
    time_scale = turbulence_length / velocity
    exp_factor = np.exp(-dt / time_scale) * np.ones_like(flow_rate)
    eta = np.sqrt(1 - exp_factor**2) * np.random.normal(size=flow_rate.shape)
    new_perturbuation = old_perturbuation * exp_factor + eta * turbulence_int
    sigma = turbulence_length / dx
    return gaussian_filter1d(new_perturbuation, sigma, mode='wrap')


