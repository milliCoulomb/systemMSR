# main.py
import numpy as np
import os
import logging
from parsers.input_parser import InputDeck
from physics.neutronics import NeutronicsSolver
import matplotlib.pyplot as plt
from utils.initializer import initialize_simulation
from couplers.SteadyStateCoupler import SteadyStateCoupler
from couplers.UnsteadyCoupler import UnsteadyCoupler
from physics.turbulence import Re, velocity_calculator, in_core_time, out_core_time

# MAGIC CONSTANTS
NUMBER_OF_RENORMALIZATION_ITERATIONS = 20

def setup_logging():
    """Configure logging to file and console."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "simulation.log")
    
    # Define logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture all levels
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite log file each run
            logging.StreamHandler()  # Also output to console
        ]
    )


def main():
    setup_logging()
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
                simulation_objects["neut_solver"].n_cells
            )
            neut_solver = simulation_objects["neut_solver"]
            core_state = simulation_objects["core_state"]
            secondary_state = simulation_objects["secondary_state"]
            th_params_primary = simulation_objects["th_params_primary"]
            # calculate the reynolds number
            reynolds_number = Re(
                simulation_objects["time_params"].pump_values_primary,
                simulation_objects["core_geom"].core_radius,
                simulation_objects["core_geom"].core_length,
                simulation_objects["th_params_primary"].rho,
                simulation_objects["th_params_primary"].mu,
            )
            print(f"Reynolds number: {reynolds_number}")
            # calculate the velocity
            velocity = velocity_calculator(
                simulation_objects["time_params"].pump_values_primary,
                simulation_objects["core_geom"].core_radius,
                simulation_objects["th_params_primary"].rho,
            )
            print(f"Velocity: {velocity}")
            # calculate the time it takes for the fluid to pass through the core
            core_time = in_core_time(
                simulation_objects["time_params"].pump_values_primary,
                simulation_objects["core_geom"].core_radius,
                simulation_objects["core_geom"].core_length,
                simulation_objects["th_params_primary"].rho,
            )
            print(f"Time to pass through the core: {core_time}")
            # calculate the out of core time
            out_core_t = out_core_time(
                simulation_objects["time_params"].pump_values_primary,
                simulation_objects["core_geom"].core_radius,
                simulation_objects["core_geom"].exchanger_length,
                simulation_objects["th_params_primary"].rho,
            )
            print(f"Time to pass through the out of core region: {out_core_t}")
            th_solver = simulation_objects["th_solver"]
            # solve the uncoupled problem
            initial_neutronics_state = neut_solver.solve_static(
                th_state=core_state,
                th_params=th_params_primary,
                source=source,
                override_mode="criticality",
            )
            coupler = SteadyStateCoupler(
                th_solver=th_solver,
                neutronics_solver=neut_solver,
                mode="criticality",
            )
            (
                coupled_core_state,
                coupled_secondary_state,
                final_state,
            ) = coupler.solve(
                th_state_primary=core_state,
                th_state_secondary=secondary_state,
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

            plt.plot(x, coupled_core_state.temperature, label="Core")
            plt.xlabel("Position [m]")
            plt.ylabel("Temperature [K]")
            plt.show()
            plt.close()

            plt.plot(
                x_secondary,
                coupled_secondary_state.temperature,
                label="Secondary",
            )
            plt.xlabel("Position [m]")
            plt.ylabel("Temperature [K]")
            plt.show()
            plt.close()

            # print the final keff
            print(f"Final keff: {final_state.keff}")
        elif simulation_objects["neutronic_mode"] == "source_driven":
            # we first override the mode of the neutronics solver and check if the uncoupled problem is solvable
            beam_center = simulation_objects["time_params"].accelerator_center
            beam_width = simulation_objects["time_params"].accelerator_width
            beam_intensity = simulation_objects["time_params"].accelerator_intensity_values[0]
            # source is a Gaussian distribution of the beam intensity centered at the beam center with a width of the beam width
            x = np.linspace(
                0,
                simulation_objects["core_geom"].core_length
                + simulation_objects["core_geom"].exchanger_length,
                simulation_objects["core_geom"].n_cells_core + simulation_objects["core_geom"].n_cells_exchanger,
            )

            source = beam_intensity * np.exp(
                -((x - beam_center) ** 2)
                / (2 * beam_width ** 2)
            )

            # plot the source
            plt.plot(x, source)
            plt.xlabel("Position [m]")
            plt.ylabel("Gamma flux [n/m^2-s]")
            plt.title("Source Distribution")
            plt.show()
            plt.close()
            
            initial_neutronics_state = simulation_objects["neut_solver"].solve_static(
                th_state=simulation_objects["core_state"],
                th_params=simulation_objects["th_params_primary"],
                source=source,
                override_mode="criticality",
            )
            print(f"Initial keff: {initial_neutronics_state.keff}")
            if initial_neutronics_state.keff > 1.0:
                raise ValueError(
                    "The source driven mode is not solvable for the given problem (keff > 1.0)"
                )
            coupler = SteadyStateCoupler(
                simulation_objects["th_solver"],
                simulation_objects["neut_solver"],
                simulation_objects["neutronic_mode"],
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

            print(f"Final power: {final_state.power / 1e6} MW")
        else:
            raise ValueError("Invalid neutronics mode")
    elif simulation_objects["simulation_mode"] == "transient":
        if simulation_objects["neutronic_mode"] == "criticality":
            # print the reynolds number
            reynolds_number = Re(
                simulation_objects["time_params"].pump_values_primary,
                simulation_objects["core_geom"].core_radius,
                simulation_objects["core_geom"].core_length,
                simulation_objects["th_params_primary"].rho,
                simulation_objects["th_params_primary"].mu,
            )
            print(f"Reynolds number: {reynolds_number}")
            input("Press Enter to continue...")
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
            input("Press Enter to continue...")
            # solve the unsteady coupled problem
            unsteady_coupler = UnsteadyCoupler(
                th_solver=simulation_objects["th_solver"],
                neutronics_solver=simulation_objects["neut_solver"],
                initial_neutronics_state=final_state,
                initial_th_state_primary=simulation_objects["core_state"],
                initial_th_state_secondary=simulation_objects["secondary_state"],
                operational_parameters=simulation_objects["time_params"],
                turbulence=simulation_objects["turbulence_bool"],
            )

            core_states, secondary_states, neutronic_states = unsteady_coupler.solve()
            index_to_start = (
                np.abs(simulation_objects["time_params"].time_values - 0.0)
            ).argmin()
            time = simulation_objects["time_params"].time_values[index_to_start:]
            power = np.array([state.power for state in neutronic_states])[
                index_to_start:
            ]
            flow_rate_secondary = np.array(
                [state.flow_rate for state in secondary_states]
            )[index_to_start:][:, 0]
            # is an array of size (n_cells, n_time_steps)
            flow_rate_primary = np.array([state.flow_rate for state in core_states])[
                index_to_start:
            ][:, 0]

            # Plotting time series data (optional)
            nominal_power = simulation_objects["neut_params"].power
            fig, ax1 = plt.subplots()
            color = "tab:red"
            ax1.set_xlabel("Time [s]")
            # show in percentage of nominal power
            ax1.set_ylabel("Power [%]", color=color)
            ax1.plot(time, power / nominal_power * 100, color=color, label="Power")
            ax1.tick_params(axis="y", labelcolor=color)

            # ax2 = ax1.twinx()
            # color = "tab:blue"
            # ax2.set_ylabel("Flow Rate [kg/s]", color=color)
            # ax2.plot(
            #     time, flow_rate_secondary, color="tab:blue", label="Secondary Flow Rate"
            # )
            # ax2.plot(
            #     time, flow_rate_primary, color="tab:green", label="Primary Flow Rate"
            # )
            # ax2.tick_params(axis="y", labelcolor=color)

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

            # save the power and temperature data with the delayed fraction in the name
            delayed_fraction = simulation_objects["neut_params"].beta
            np.save(
                f"power_temperature_data_{delayed_fraction}.npy",
                (time, power, out_temp, in_temp),
            )


        elif simulation_objects["neutronic_mode"] == "source_driven":
            # we first override the mode of the neutronics solver and check if the uncoupled problem is solvable
            beam_center = simulation_objects["time_params"].accelerator_center
            beam_width = simulation_objects["time_params"].accelerator_width
            beam_intensity = simulation_objects["time_params"].accelerator_intensity_values[0]
            # source is a Gaussian distribution of the beam intensity centered at the beam center with a width of the beam width
            x = np.linspace(
                0,
                simulation_objects["core_geom"].core_length
                + simulation_objects["core_geom"].exchanger_length,
                simulation_objects["core_geom"].n_cells_core + simulation_objects["core_geom"].n_cells_exchanger,
            )

            source = beam_intensity * np.exp(
                -((x - beam_center) ** 2)
                / (2 * beam_width ** 2)
            )

            # plot the source
            plt.plot(x, source)
            plt.xlabel("Position [m]")
            plt.ylabel("Gamma flux [n/m^2-s]")
            plt.title("Source Distribution")
            plt.show()
            plt.close()
            
            initial_neutronics_state = simulation_objects["neut_solver"].solve_static(
                th_state=simulation_objects["core_state"],
                th_params=simulation_objects["th_params_primary"],
                source=source,
                override_mode="criticality",
            )
            print(f"Initial keff: {initial_neutronics_state.keff}")
            if initial_neutronics_state.keff > 1.0:
                raise ValueError(
                    "The source driven mode is not solvable for the given problem (keff > 1.0)"
                )
            coupler = SteadyStateCoupler(
                simulation_objects["th_solver"],
                simulation_objects["neut_solver"],
                simulation_objects["neutronic_mode"],
            )
            (
                coupled_core_state,
                coupled_secondary_state,
                coupled_neutronic_state,
            ) = coupler.solve(
                simulation_objects["core_state"],
                simulation_objects["secondary_state"],
                initial_neutronics_state=initial_neutronics_state,
                source=source,
            )
            if coupled_neutronic_state.keff > 1.0:
                raise ValueError(
                    "The source driven mode is not solvable for the given problem (keff > 1.0)"
                )
            print("Steady state source driven problem solved, starting transient simulation")
            print(f"Initial keff: {coupled_neutronic_state.keff}")
            print(f"Initial power: {coupled_neutronic_state.power / 1e6} MW")
            input("Press Enter to continue...")
            # solve the unsteady coupled problem
            unsteady_coupler = UnsteadyCoupler(
                th_solver=simulation_objects["th_solver"],
                neutronics_solver=simulation_objects["neut_solver"],
                initial_neutronics_state=coupled_neutronic_state,
                initial_th_state_primary=coupled_core_state,
                initial_th_state_secondary=coupled_secondary_state,
                operational_parameters=simulation_objects["time_params"],
                turbulence=simulation_objects["turbulence_bool"],
            )
            core_states, secondary_states, neutronic_states = unsteady_coupler.solve()

            index_to_start = 0
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
            accelerator_intensity = np.array(
                simulation_objects["time_params"].accelerator_intensity_values
            )[index_to_start:]

            # Plotting time series data (optional)
            fig, ax1 = plt.subplots()
            color = "tab:red"
            ax1.set_xlabel("Time [s]")
            # show in percentage of nominal power
            ax1.set_ylabel("Power [MW]", color=color)
            ax1.plot(time, power / 1e6, color=color, label="Power")
            ax1.tick_params(axis="y", labelcolor=color)
            
            ax2 = ax1.twinx()
            color = "tab:blue"
            ax2.set_ylabel("Accelerator Intensity [gamma/m2/s]", color=color)
            ax2.plot(time, accelerator_intensity, color=color, label="Intensity")
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
        else:
            raise ValueError("Invalid neutronics mode")
            


if __name__ == "__main__":
    main()
