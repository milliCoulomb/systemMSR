# main.py
import numpy as np
from parsers.input_parser import InputDeck
from physics.neutronics import NeutronicsSolver
from physics.thermo import ThermoHydraulicsSolver
import matplotlib.pyplot as plt
from utils.initializer import initialize_simulation
from couplers.SteadyStateCoupler import SteadyStateCoupler
from couplers.UnsteadyCoupler import UnsteadyCoupler


def main():
    # Path to your input deck
    input_deck_path = "input/input_deck.yaml"

    # Parse the input deck
    input_deck = InputDeck.from_yaml(input_deck_path)

    simulation_objects = initialize_simulation(input_deck)
    
    # solve the neutronics problem
    final_state = simulation_objects["neut_solver"].solve_static(simulation_objects["core_state"], simulation_objects["th_params_primary"])
    # plot the final neutron flux
    x = np.linspace(
        0, simulation_objects["core_geom"].core_length + simulation_objects["core_geom"].exchanger_length, len(final_state.phi)
    )
    print(f"keff: {final_state.keff}")
    # renormalize the fission cross section by the keff
    simulation_objects["neut_params"].Sigma_f = simulation_objects["neut_params"].Sigma_f / final_state.keff
    # solve the thermo-hydraulics problem
    th_solver = ThermoHydraulicsSolver(
        simulation_objects["th_params_primary"],
        simulation_objects["th_params_secondary"],
        simulation_objects["fvm"],
        simulation_objects["core_geom"],
        simulation_objects["fvm_secondary"],
        simulation_objects["secondary_geom"],
    )
    simulation_objects["core_state"], simulation_objects["secondary_state"] = th_solver.solve_static(
        simulation_objects["core_state"], simulation_objects["secondary_state"], final_state
    )

    x_secondary = np.linspace(
        0,
        simulation_objects["secondary_geom"].first_loop_length
        + simulation_objects["secondary_geom"].exchanger_length
        + simulation_objects["secondary_geom"].second_loop_length,
        len(simulation_objects["secondary_state"].temperature),
    )

    # solve the coupled problem
    number_of_iterations = 20
    for _ in range(number_of_iterations):
        coupler = SteadyStateCoupler(th_solver, simulation_objects["neut_solver"])
        simulation_objects["core_state"], simulation_objects["secondary_state"], neutronic_state = coupler.solve(
            simulation_objects["core_state"], simulation_objects["secondary_state"], final_state
        )
        # print the coupled keff
        print(f"Intermediate keff: {neutronic_state.keff}")
        # renormalize the fission cross section by the keff
        simulation_objects["neut_params"].Sigma_f = simulation_objects["neut_params"].Sigma_f / neutronic_state.keff
        # rebuild the neutronics solver
        simulation_objects["neut_solver"] = NeutronicsSolver(simulation_objects["neut_params"], simulation_objects["fvm"], simulation_objects["core_geom"])
    
    # Final keff after the loop
    print(f"Final keff: {neutronic_state.keff}")
    # plot the final temperature distribution
    plt.plot(x, simulation_objects["core_state"].temperature, label="Core")
    plt.xlabel("Position [m]")
    plt.ylabel("Temperature [K]")
    plt.show()
    plt.close()

    plt.plot(x_secondary, simulation_objects["secondary_state"].temperature, label="Secondary")
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

    # solve the unsteady coupled problem
    unsteady_coupler = UnsteadyCoupler(
        th_solver,
        simulation_objects["neut_solver"],
        neutronic_state,
        simulation_objects["core_state"],
        simulation_objects["secondary_state"],
        simulation_objects["time_params"],
    )

    core_states, secondary_states, neutronic_states = unsteady_coupler.solve()
    index_to_start = (np.abs(simulation_objects["time_params"].time_values - 20.0)).argmin()
    time = simulation_objects["time_params"].time_values[index_to_start:]
    power = np.array([state.power for state in neutronic_states])[index_to_start:]
    flow_rate_secondary = np.array([state.flow_rate for state in secondary_states])[index_to_start:]
    flow_rate_primary = np.array([state.flow_rate for state in core_states])[index_to_start:]

    # Plotting time series data (optional)
    nominal_power = simulation_objects["neut_params"].power
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Time [s]")
    # show in percentage of nominal power
    ax1.set_ylabel("Power [%]", color=color)
    ax1.plot(time, power / nominal_power * 100, color=color, label="Power")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Flow Rate [kg/s]", color=color)
    ax2.plot(time, flow_rate_secondary, color="tab:blue", label="Secondary Flow Rate")
    ax2.plot(time, flow_rate_primary, color="tab:green", label="Primary Flow Rate")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.legend(loc='upper left')
    plt.show()

    n_core = 300
    in_core = 0
    out_temp = np.array([state.temperature[n_core] for state in core_states])[index_to_start:]
    in_temp = np.array([state.temperature[in_core] for state in core_states])[index_to_start:]
    plt.plot(time, out_temp, label="Outlet Core Temperature")
    plt.plot(time, in_temp, label="Inlet Core Temperature")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [K]")
    plt.legend()
    plt.show()
    plt.close()  # Ensure this is commented out


if __name__ == "__main__":
    main()
