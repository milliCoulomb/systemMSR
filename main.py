# main.py
import numpy as np
from parsers.input_parser import InputDeck
from physics.neutronics import NeutronicsSolver, NeutronicsParameters
from physics.thermo import ThermoHydraulicsParameters, ThermoHydraulicsSolver
from methods.fvm import FVM
from utils.geometry import CoreGeometry
from physics.thermo import ThermoHydraulicsParameters
import matplotlib.pyplot as plt
from utils.states import ThermoHydraulicsState, NeutronicsState


def main():
    # Path to your input deck
    input_deck_path = "input/input_deck.yaml"

    # Parse the input deck
    input_deck = InputDeck.from_yaml(input_deck_path)

    # Accessing some parameters as examples
    print(f"Total Simulation Time: {input_deck.simulation.total_time} s")
    print(f"Time Step: {input_deck.simulation.time_step} s")

    print(f"Core Length: {input_deck.geometry.core_length} m")
    print(
        f"Heat Exchanger Coefficient: {input_deck.geometry.heat_exchanger_coefficient} W/m^3-K"
    )

    print(
        f"Primary Salt Density: {input_deck.materials.primary_salt['density']} kg/m^3"
    )
    print(f"Secondary Salt CP: {input_deck.materials.secondary_salt['cp']} J/kg-K")

    print(
        f"Nuclear Diffusion Coefficient: {input_deck.nuclear_data.diffusion_coefficient} m"
    )

    # Example: Access pump primary schedule
    print("Primary Pump Schedule:")
    for point in input_deck.operational_parameters.pump_primary.schedule:
        print(f"  Time: {point.time} s, Flow Rate: {point.flow_rate} kg/s")

    core_geom = CoreGeometry(
        core_length=input_deck.geometry.core_length,
        exchanger_length=input_deck.geometry.exchanger_length,
        core_radius=input_deck.geometry.core_radius,
        n_cells_core=input_deck.geometry.n_core,
        n_cells_exchanger=input_deck.geometry.n_exchanger,
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
    )
    fvm = FVM(core_geom.dx)
    neut_solver = NeutronicsSolver(neut_params, fvm, core_geom)
    # initial temperature distribution
    core_state = ThermoHydraulicsState(
        temperature=np.ones_like(core_geom.dx) * 900.0,
        flow_rate=10.0,
        T_in=0.0,
    )
    secondary_state = ThermoHydraulicsState(
        temperature=np.ones_like(core_geom.dx) * 900.0,
        flow_rate=10.0,
        T_in=920.0,
    )
    velocity = core_state.flow_rate / (input_deck.materials.primary_salt["density"] * np.pi * core_geom.core_radius ** 2)
    print(f"Velocity: {velocity}")
    # initiate the thermo-hydraulics parameters
    th_params = ThermoHydraulicsParameters(
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
    final_state = neut_solver.solve_static(core_state, th_params)
    # plot the final neutron flux
    x = np.linspace(0, core_geom.core_length + core_geom.exchanger_length, len(final_state.phi))
    plt.plot(x, final_state.phi)
    plt.xlabel("Position [m]")
    plt.ylabel("Neutron Flux [n/m^2-s]")
    plt.title("Neutron Flux Distribution")
    plt.show()
    plt.close()
    # plot precursor concentration
    plt.plot(x, final_state.C)
    plt.xlabel("Position [m]")
    plt.ylabel("Precursor Concentration [n/m^3]")
    plt.title("Precursor Concentration Distribution")
    plt.show()
    print(f'keff: {final_state.keff}')






if __name__ == "__main__":
    main()
