# utils/initializer.py

import numpy as np
from utils.geometry import CoreGeometry, SecondaryLoopGeometry
from physics.neutronics import NeutronicsParameters
from methods.fvm import FVM
from physics.neutronics import NeutronicsSolver
from utils.states import ThermoHydraulicsState
from utils.time_parameters import TimeParameters

def initialize_simulation(input_deck):
    # Print simulation parameters
    print(f"Total Simulation Time: {input_deck.simulation.total_time} s")
    print(f"Time Step: {input_deck.simulation.time_step} s")
    
    print(f"Core Length: {input_deck.geometry.core_length} m")
    print(f"Heat Exchanger Coefficient: {input_deck.geometry.heat_exchanger_coefficient} W/m^3-K")
    
    print(f"Primary Salt Density: {input_deck.materials.primary_salt['density']} kg/m^3")
    print(f"Secondary Salt CP: {input_deck.materials.secondary_salt['cp']} J/kg-K")
    
    print(f"Nuclear Diffusion Coefficient: {input_deck.nuclear_data.diffusion_coefficient} m")
    
    # Initialize pump schedules
    pump_primary = initialize_pump_schedule("Primary", input_deck)
    pump_secondary = initialize_pump_schedule("Secondary", input_deck)
    
    # Initialize secondary inlet temperature schedule
    times_sec_in_temp, temps_sec_in_temp = initialize_secondary_inlet_temperature(input_deck)
    print("Secondary Inlet Temperature Schedule:")
    print(f"Times: {times_sec_in_temp}")
    print(f"Temperatures: {temps_sec_in_temp}")
    
    # Initialize geometries
    core_geom = CoreGeometry(
        core_length=input_deck.geometry.core_length,
        exchanger_length=input_deck.geometry.exchanger_length,
        core_radius=input_deck.geometry.core_radius,
        n_cells_core=input_deck.geometry.n_core,
        n_cells_exchanger=input_deck.geometry.n_exchanger,
    )
    secondary_geom = SecondaryLoopGeometry(
        first_loop_length=input_deck.geometry.cooling_loop_first_length,
        exchanger_length=input_deck.geometry.exchanger_length,
        second_loop_length=input_deck.geometry.cooling_loop_second_length,
        loop_radius=input_deck.geometry.secondary_loop_radius,
        n_cells_first_loop=input_deck.geometry.n_cooling_loop_first_segment,
        n_cells_exchanger=input_deck.geometry.n_exchanger,
        n_cells_second_loop=input_deck.geometry.n_cooling_loop_second_segment,
    )
    
    # Initialize neutronics parameters
    nuc = input_deck.nuclear_data
    neut_params = NeutronicsParameters(
        D=nuc.diffusion_coefficient,
        Sigma_a=nuc.absorption_cross_section,
        Sigma_f=nuc.fission_cross_section,
        nu_fission=nuc.nu_fission,
        beta=nuc.beta,
        Lambda=nuc.decay_constant,
        kappa=nuc.kappa,
        power=nuc.power,
        neutron_velocity=nuc.neutron_velocity,
    )
    
    # Initialize solvers
    fvm = FVM(core_geom.dx)
    fvm_secondary = FVM(secondary_geom.dx)
    neut_solver = NeutronicsSolver(neut_params, fvm, core_geom)
    
    # Initialize thermohydraulics states
    core_state = ThermoHydraulicsState(
        temperature=np.ones_like(core_geom.dx) * 900.0,
        flow_rate=pump_primary['rates'][0],
        T_in=0.0,
    )
    secondary_state = ThermoHydraulicsState(
        temperature=np.ones_like(secondary_geom.dx) * 900.0,
        flow_rate=pump_secondary['rates'][0],
        T_in=temps_sec_in_temp[0],
    )
    
    velocity = core_state.flow_rate / (
        input_deck.materials.primary_salt["density"] * np.pi * core_geom.core_radius**2
    )
    print(f"Velocity: {velocity}")
    
    # Initialize time parameters
    NUMBER_TIME_STEPS = int(input_deck.simulation.total_time / input_deck.simulation.time_step)
    time_params = TimeParameters(
        time_step=input_deck.simulation.time_step,
        total_time=input_deck.simulation.total_time,
        num_time_steps=NUMBER_TIME_STEPS,
        pump_values_primary=pump_primary['rates'],
        time_values_primary_pump=pump_primary['times'],
        pump_values_secondary=pump_secondary['rates'],
        time_values_secondary_pump=pump_secondary['times'],
        secondary_inlet_temperature_values=temps_sec_in_temp,
        time_values_secondary_inlet_temperature=times_sec_in_temp,
    )
    
    return {
        "core_geom": core_geom,
        "secondary_geom": secondary_geom,
        "neut_params": neut_params,
        "fvm": fvm,
        "fvm_secondary": fvm_secondary,
        "neut_solver": neut_solver,
        "core_state": core_state,
        "secondary_state": secondary_state,
        "velocity": velocity,
        "time_params": time_params,
    }

def initialize_pump_schedule(pump_type, input_deck):
    if pump_type.lower() == "primary":
        schedule = input_deck.operational_parameters.pump_primary.schedule
    elif pump_type.lower() == "secondary":
        schedule = input_deck.operational_parameters.pump_secondary.schedule
    else:
        raise ValueError("Pump type must be 'Primary' or 'Secondary'")
    
    times = np.array([point.time for point in schedule])
    rates = np.array([point.flow_rate for point in schedule])
    
    print(f"{pump_type} Pump Schedule:")
    print(f"Times: {times}")
    print(f"Flow Rates: {rates}")
    
    return {"times": times, "rates": rates}

def initialize_secondary_inlet_temperature(input_deck):
    schedule = input_deck.operational_parameters.secondary_inlet_temp.schedule
    times = np.array([point.time for point in schedule])
    temperatures = np.array([point.temperature for point in schedule])
    
    return times, temperatures