# main.py
import numpy as np
from parsers.input_parser import InputDeck
from physics.neutronics import NeutronicsSolver
from physics.thermo import ThermoHydraulicsSolver
import matplotlib.pyplot as plt
from utils.initializer import initialize_simulation
from couplers.SteadyStateCoupler import SteadyStateCoupler
from couplers.UnsteadyCoupler import UnsteadyCoupler

# MAGIC CONSTANTS
NUMBER_OF_RENORMALIZATION_ITERATIONS = 20


def main():
    # Path to your input deck
    input_deck_path = "input/input_deck.yaml"

    # Parse the input deck
    input_deck = InputDeck.from_yaml(input_deck_path)

    simulation_objects = initialize_simulation(input_deck)

    # first thing first is to solve the steady state criticality problem without coupling

    # check type of simulation and then the neutronic mode
    if simulation_objects["simulation_mode"] == "steady_state":
        if simulation_objects["neutronic_mode"] == "criticality":
            # solve the steady state criticality problem
            source = np.zeros(
                simulation_objects["core_geom"].n_core
                + simulation_objects["core_geom"].n_exchanger
            )
            # solve the uncoupled problem
            initial_neutronics_state = simulation_objects["neut_solver"].solve_static(
                th_state=simulation_objects["core_state"],
                th_params=simulation_objects["th_params_primary"],
                source=source,
                override_mode="criticality",
            )
            coupler = SteadyStateCoupler(
                simulation_objects["th_solver"],
                simulation_objects["neut_solver"],
                simulation_objects["neutronics_mode"],
            )
            (
                simulation_objects["core_state"],
                simulation_objects["secondary_state"],
                final_state,
            ) = coupler.solve(
                simulation_objects["core_state"],
                simulation_objects["secondary_state"],
                initial_neutronics_state=initial_neutronics_state,
                source=source,
            )
            # plot the final states
            x = np.linspace(
                0,
                simulation_objects["core_geom"].core_length
                + simulation_objects["core_geom"].exchanger_length,
                len(final_state.phi),
            )
            plt.plot(x, final_state.phi)
            plt.xlabel("Position [m]")
            plt.ylabel("Neutron Flux [n/m^2-s]")
            plt.title("Neutron Flux Distribution")
            plt.show()
            plt.close()

            x_secondary = np.linspace(
                0,
                simulation_objects["secondary_geom"].first_loop_length
                + simulation_objects["secondary_geom"].exchanger_length
                + simulation_objects["secondary_geom"].second_loop_length,
                len(simulation_objects["secondary_state"].temperature),
            )

            plt.plot(x, simulation_objects["core_state"].temperature, label="Core")
            plt.xlabel("Position [m]")
            plt.ylabel("Temperature [K]")
            plt.show()
            plt.close()

            plt.plot(
                x_secondary,
                simulation_objects["secondary_state"].temperature,
                label="Secondary",
            )
            plt.xlabel("Position [m]")
            plt.ylabel("Temperature [K]")
            plt.show()
            plt.close()

            # print the final keff
            print(f"Final keff: {final_state.keff}")
        elif simulation_objects["neutronics_mode"] == "source_driven":
            # in this case, we take the first source value in time_params, it is a Dirac centered at n_core // 2
            source = np.zeros(
                simulation_objects["core_geom"].n_core
                + simulation_objects["core_geom"].n_exchanger
            )
            source[simulation_objects["core_geom"].n_core // 2] = simulation_objects[
                "time_params"
            ].accelerator_intensity_values[0]
            coupler = SteadyStateCoupler(
                simulation_objects["th_solver"],
                simulation_objects["neut_solver"],
                simulation_objects["neutronics_mode"],
            )
            (
                simulation_objects["core_state"],
                simulation_objects["secondary_state"],
                final_state,
            ) = coupler.solve(
                simulation_objects["core_state"],
                simulation_objects["secondary_state"],
                simulation_objects["neut_params"],
                source=source,
            )
            # plot the final states
            x = np.linspace(
                0,
                simulation_objects["core_geom"].core_length
                + simulation_objects["core_geom"].exchanger_length,
                len(final_state.phi),
            )
            plt.plot(x, final_state.phi)
            plt.xlabel("Position [m]")
            plt.ylabel("Neutron Flux [n/m^2-s]")
            plt.title("Neutron Flux Distribution")
            plt.show()
            plt.close()

            x_secondary = np.linspace(
                0,
                simulation_objects["secondary_geom"].first_loop_length
                + simulation_objects["secondary_geom"].exchanger_length
                + simulation_objects["secondary_geom"].second_loop_length,
                len(simulation_objects["secondary_state"].temperature),
            )

            plt.plot(x, simulation_objects["core_state"].temperature, label="Core")
            plt.xlabel("Position [m]")
            plt.ylabel("Temperature [K]")
            plt.show()

            plt.plot(
                x_secondary,
                simulation_objects["secondary_state"].temperature,
                label="Secondary",
            )
            plt.xlabel("Position [m]")
            plt.ylabel("Temperature [K]")
            plt.show()
            plt.close()
        else:
            raise ValueError("Invalid neutronics mode")
    elif simulation_objects["simulation_mode"] == "transient":
        if simulation_objects["neutronic_mode"] == "criticality":
            coupler = SteadyStateCoupler(
                simulation_objects["th_solver"],
                simulation_objects["neut_solver"],
                simulation_objects["neutronic_mode"],
            )
            # solve the steady state criticality problem and renormalize the fission cross section
            source = np.zeros(
                simulation_objects["core_geom"].n_cells_core
                + simulation_objects["core_geom"].n_cells_exchanger
            )
            # solve the uncoupled problem
            initial_neutronics_state = simulation_objects["neut_solver"].solve_static(
                th_state=simulation_objects["core_state"],
                th_params=simulation_objects["th_params_primary"],
                source=source,
                override_mode="criticality",
            )
            for _ in range(NUMBER_OF_RENORMALIZATION_ITERATIONS):
                (
                    simulation_objects["core_state"],
                    simulation_objects["secondary_state"],
                    final_state,
                ) = coupler.solve(
                    simulation_objects["core_state"],
                    simulation_objects["secondary_state"],
                    initial_neutronics_state=initial_neutronics_state,
                    source=source,
                )
                # renormalize the fission cross section by the keff
                simulation_objects["neut_params"].Sigma_f = (
                    simulation_objects["neut_params"].Sigma_f / final_state.keff
                )
                # rebuild the neutronics solver
                simulation_objects["neut_solver"] = NeutronicsSolver(
                    simulation_objects["neut_params"],
                    simulation_objects["fvm"],
                    simulation_objects["core_geom"],
                    mode=simulation_objects["neutronic_mode"],
                )
            # print the final keff
            print(f"Final keff after renormalization: {final_state.keff}")
            # solve the unsteady coupled problem
            unsteady_coupler = UnsteadyCoupler(
                th_solver=simulation_objects["th_solver"],
                neutronics_solver=simulation_objects["neut_solver"],
                initial_neutronics_state=final_state,
                initial_th_state_primary=simulation_objects["core_state"],
                initial_th_state_secondary=simulation_objects["secondary_state"],
                operational_parameters=simulation_objects["time_params"],
            )

            core_states, secondary_states, neutronic_states = unsteady_coupler.solve()
            index_to_start = (
                np.abs(simulation_objects["time_params"].time_values - 20.0)
            ).argmin()
            time = simulation_objects["time_params"].time_values[index_to_start:]
            power = np.array([state.power for state in neutronic_states])[
                index_to_start:
            ]
            flow_rate_secondary = np.array(
                [state.flow_rate for state in secondary_states]
            )[index_to_start:]
            flow_rate_primary = np.array([state.flow_rate for state in core_states])[
                index_to_start:
            ]

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
            ax2.plot(
                time, flow_rate_secondary, color="tab:blue", label="Secondary Flow Rate"
            )
            ax2.plot(
                time, flow_rate_primary, color="tab:green", label="Primary Flow Rate"
            )
            ax2.tick_params(axis="y", labelcolor=color)

            fig.tight_layout()
            plt.legend(loc="upper left")
            plt.show()
            plt.close()

            n_core = 300
            in_core = 0
            out_temp = np.array([state.temperature[n_core] for state in core_states])[
                index_to_start:
            ]
            in_temp = np.array([state.temperature[in_core] for state in core_states])[
                index_to_start:
            ]
            plt.plot(time, out_temp, label="Outlet Core Temperature")
            plt.plot(time, in_temp, label="Inlet Core Temperature")
            plt.xlabel("Time [s]")
            plt.ylabel("Temperature [K]")
            plt.legend()
            plt.show()
            plt.close()
        elif simulation_objects["neutronics_mode"] == "source_driven":
            raise ValueError("Transient mode not implemented for source driven mode")


if __name__ == "__main__":
    main()
