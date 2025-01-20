# main.py
import numpy as np
from parsers.input_parser import InputDeck
from physics.neutronics import NeutronicsSolver, NeutronicsParameters
from physics.thermo import ThermoHydraulicsParameters, ThermoHydraulicsSolver
from methods.fvm import FVM
from utils.geometry import CoreGeometry, SecondaryLoopGeometry
from physics.thermo import ThermoHydraulicsParameters
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.states import ThermoHydraulicsState, NeutronicsState
from utils.time_parameters import TimeParameters
from utils.initializer import initialize_simulation
from couplers.SteadyStateCoupler import SteadyStateCoupler
from couplers.UnsteadyCoupler import UnsteadyCoupler


def main():
    # Path to your input deck
    input_deck_path = "input/input_deck.yaml"

    # Parse the input deck
    input_deck = InputDeck.from_yaml(input_deck_path)

    simulation_objects = initialize_simulation(input_deck)

    # Unpack simulation objects
    core_geom = simulation_objects["core_geom"]
    secondary_geom = simulation_objects["secondary_geom"]
    neut_params = simulation_objects["neut_params"]
    fvm = simulation_objects["fvm"]
    fvm_secondary = simulation_objects["fvm_secondary"]
    neut_solver = simulation_objects["neut_solver"]
    core_state = simulation_objects["core_state"]
    secondary_state = simulation_objects["secondary_state"]
    velocity = simulation_objects["velocity"]
    time_params = simulation_objects["time_params"]
    th_params_primary = simulation_objects["th_params_primary"]
    th_params_secondary = simulation_objects["th_params_secondary"]
    
    # solve the neutronics problem
    final_state = neut_solver.solve_static(core_state, th_params_primary)
    # plot the final neutron flux
    x = np.linspace(
        0, core_geom.core_length + core_geom.exchanger_length, len(final_state.phi)
    )
    print(f"keff: {final_state.keff}")
    # renormalize the fission cross section by the keff
    neut_params.Sigma_f = neut_params.Sigma_f / final_state.keff
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
    number_of_iterations = 20
    for _ in range(number_of_iterations):
        coupler = SteadyStateCoupler(th_solver, neut_solver)
        core_state, secondary_state, neutronic_state = coupler.solve(
            core_state, secondary_state, final_state
        )
        # print the coupled keff
        print(f"Intermediate keff: {neutronic_state.keff}")
        # renormalize the fission cross section by the keff
        neut_params.Sigma_f = neut_params.Sigma_f / neutronic_state.keff
        # rebuild the neutronics solver
        neut_solver = NeutronicsSolver(neut_params, fvm, core_geom)
    
    # Final keff after the loop
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

    # solve the unsteady coupled problem
    unsteady_coupler = UnsteadyCoupler(
        th_solver,
        neut_solver,
        neutronic_state,
        core_state,
        secondary_state,
        time_params,
    )

    core_states, secondary_states, neutronic_states = unsteady_coupler.solve()
    index_to_start = (np.abs(time_params.time_values - 0.1)).argmin()
    time = time_params.time_values[index_to_start:]
    power = np.array([state.power for state in neutronic_states])[index_to_start:]
    flow_rate_secondary = np.array([state.flow_rate for state in secondary_states])[index_to_start:]
    flow_rate_primary = np.array([state.flow_rate for state in core_states])[index_to_start:]

    # Plotting time series data (optional)
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Power [MW]", color=color)
    ax1.plot(time, power / 1e6, color=color, label="Power")
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
    # plt.close()  # Ensure this is commented out

    # plot the max temperature in the core
    max_temp = np.array([np.max(state.temperature) for state in core_states])[index_to_start:]
    plt.plot(time, max_temp, label="Max Temperature")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [K]")
    plt.legend()
    plt.show()
    plt.close()  # Ensure this is commented out

    # # Animation section
    # fig, ax = plt.subplots()
    # line, = ax.plot([], [], lw=2, color='tab:red')

    # # Precompute x
    # x = np.linspace(0, core_geom.core_length, len(core_states[index_to_start].temperature))

    # # Compute min and max temperatures for y-axis limits
    # min_temp = np.min([np.min(state.temperature) for state in core_states[index_to_start:]])
    # max_temp = np.max([np.max(state.temperature) for state in core_states[index_to_start:]])

    # ax.set_xlim(0, core_geom.core_length)
    # ax.set_ylim(min_temp - 10, max_temp + 10)  # Set fixed y-axis limits
    # ax.set_xlabel("Position [m]")
    # ax.set_ylabel("Temperature [K]")
    # ax.set_title("Temperature Distribution in the Core")

    # # Initialization function: plot the background of each frame
    # def init():
    #     line.set_data([], [])
    #     return (line,)

    # # Animation function. This is called sequentially
    # def animate(i):
    #     current_state = core_states[i + index_to_start]
    #     y = current_state.temperature
    #     line.set_data(x, y)
    #     return (line,)

    # # Determine number of frames
    # num_frames = len(core_states) - index_to_start
    # print(f"Number of frames for animation: {num_frames}")

    # if num_frames <= 0:
    #     raise ValueError("No frames to animate. Check 'index_to_start' and 'core_states' length.")

    # # Create the animation
    # anim = animation.FuncAnimation(
    #     fig, animate, init_func=init, frames=num_frames,
    #     interval=200, blit=False  # Set blit=False for better compatibility
    # )

    # # Optionally, save the animation to a file
    # anim.save('temperature_animation.gif', writer='pillow', fps=5)

    # plt.show()
    # plt.close()  # Ensure this is commented out


if __name__ == "__main__":
    main()
