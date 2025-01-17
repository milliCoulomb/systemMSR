# main.py
import numpy as np
from parsers.input_parser import InputDeck
from physics.neutronics import NeutronicsSolver, NeutronicsParameters
from physics.thermo import ThermoHydraulicsParameters, ThermoHydraulicsSolver
from methods.fvm import FVM
from utils.geometry import CoreGeometry, SecondaryLoopGeometry
from physics.thermo import ThermoHydraulicsParameters
import matplotlib.pyplot as plt
from utils.states import ThermoHydraulicsState, NeutronicsState
from couplers.SteadyStateCoupler import SteadyStateCoupler


def main():
    # Path to your input deck
    input_deck_path = "input/input_deck.yaml"

    # Parse the input deck
    input_deck = InputDeck.from_yaml(input_deck_path)

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
    fvm = FVM(core_geom.dx)
    fvm_secondary = FVM(secondary_geom.dx)
    neut_solver = NeutronicsSolver(neut_params, fvm, core_geom)
    # initial temperature distribution
    core_state = ThermoHydraulicsState(
        temperature=np.ones_like(core_geom.dx) * 900.0,
        flow_rate=100.0,
        T_in=0.0,
    )
    secondary_state = ThermoHydraulicsState(
        temperature=np.ones_like(core_geom.dx) * 900.0,
        flow_rate=100.0,
        T_in=800.0,
    )
    velocity = core_state.flow_rate / (
        input_deck.materials.primary_salt["density"] * np.pi * core_geom.core_radius**2
    )
    print(f"Velocity: {velocity}")
    # initiate the thermo-hydraulics parameters
    th_params_primary = ThermoHydraulicsParameters(
        rho=input_deck.materials.primary_salt["density"],
        cp=input_deck.materials.primary_salt["cp"],
        k=input_deck.materials.primary_salt["k"],
        heat_exchanger_coefficient=input_deck.geometry.heat_exchanger_coefficient,
        expansion_coefficient=input_deck.materials.primary_salt["expansion"],
    )
    th_params_secondary = ThermoHydraulicsParameters(
        rho=input_deck.materials.secondary_salt["density"],
        cp=input_deck.materials.secondary_salt["cp"],
        k=input_deck.materials.secondary_salt["k"],
        heat_exchanger_coefficient=input_deck.geometry.heat_exchanger_coefficient,
        expansion_coefficient=input_deck.materials.secondary_salt["expansion"],
    )
    # solve the neutronics problem
    final_state = neut_solver.solve_static(core_state, th_params_primary)
    # plot the final neutron flux
    x = np.linspace(
        0, core_geom.core_length + core_geom.exchanger_length, len(final_state.phi)
    )
    print(f"keff: {final_state.keff}")

    # solve the thermo-hydraulics problem
    th_solver = ThermoHydraulicsSolver(
        th_params_primary,
        th_params_secondary,
        fvm,
        core_geom,
        fvm_secondary,
        secondary_geom,
    )
    core_state, secondary_state = th_solver.solve_static(
        core_state, secondary_state, final_state
    )

    x_secondary = np.linspace(
        0,
        secondary_geom.first_loop_length
        + secondary_geom.exchanger_length
        + secondary_geom.second_loop_length,
        len(secondary_state.temperature),
    )

    # solve the coupled problem
    coupler = SteadyStateCoupler(th_solver, neut_solver)
    core_state, secondary_state, neutronic_state = coupler.solve(
        core_state, secondary_state, final_state
    )
    # print the coupled keff
    print(f"Final keff: {neutronic_state.keff}")
    # plot the final temperature distribution
    plt.plot(x, core_state.temperature, label="Core")
    plt.xlabel("Position [m]")
    plt.ylabel("Temperature [K]")
    plt.show()
    plt.close()

    plt.plot(x_secondary, secondary_state.temperature, label="Secondary")
    plt.xlabel("Position [m]")
    plt.ylabel("Temperature [K]")
    plt.show()
    plt.close()

    plt.plot(x, final_state.phi)
    plt.xlabel("Position [m]")
    plt.ylabel("Neutron Flux [n/m^2-s]")
    plt.title("Neutron Flux Distribution")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
