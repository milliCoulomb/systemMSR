# main.py
import numpy as np
import os
import logging
import argparse
from parsers.input_parser import InputDeck
from physics.neutronics import NeutronicsSolver
from utils.initializer import initialize_simulation
from utils.writer import save_post_processing
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
            logging.FileHandler(log_file, mode="w"),  # Overwrite log file each run
            logging.StreamHandler(),  # Also output to console
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the simulation.")
    parser.add_argument(
        "input_deck_path",
        type=str,
        help="Path to the input deck YAML file.",
    )
    return parser.parse_args()


def main():
    setup_logging()
    # Path to your input deck
    args = parse_args()
    input_deck_path = args.input_deck_path

    # Parse the input deck
    input_deck = InputDeck.from_yaml(input_deck_path)

    simulation_objects = initialize_simulation(input_deck)

    # first thing first is to solve the steady state criticality problem without coupling

    # check type of simulation and then the neutronic mode
    if simulation_objects["simulation_mode"] == "steady_state":
        if simulation_objects["neutronic_mode"] == "criticality":
            # solve the steady state criticality problem
            source = np.zeros(simulation_objects["neut_solver"].n_cells)
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
            core_states = [coupled_core_state]
            secondary_states = [coupled_secondary_state]
            neutronic_states = [final_state]
            # change the temperature of the core by epsilon and solve the uncoupled problem
            epsilon = 1e-2
            perturbed_core_state = core_state
            perturbed_temperature = core_state.temperature * (1 + epsilon)
            perturbed_core_state.temperature = perturbed_temperature
            perturbed_neutronics_state = neut_solver.solve_static(
                th_state=perturbed_core_state,
                th_params=th_params_primary,
                source=source,
                override_mode="criticality",
            )
            # calculate the delta T
            delta_T = epsilon * core_state.temperature
            # calculate the sensitivity of the keff to the temperature
            rho_perturbed = (1 - 1 / perturbed_neutronics_state.keff) * 1e5
            rho_initial = (1 - 1 / initial_neutronics_state.keff) * 1e5
            sensitivity = np.mean((rho_perturbed - rho_initial) / delta_T)
            print(f"Sensitivity of keff to temperature: {sensitivity} pcm/K")
            print(f"Final keff: {final_state.keff}")
        elif simulation_objects["neutronic_mode"] == "source_driven":
            # we first override the mode of the neutronics solver and check if the uncoupled problem is solvable
            beam_center = simulation_objects["time_params"].accelerator_center
            beam_width = simulation_objects["time_params"].accelerator_width
            beam_intensity = simulation_objects[
                "time_params"
            ].accelerator_intensity_values[0]
            # source is a Gaussian distribution of the beam intensity centered at the beam center with a width of the beam width
            x = np.linspace(
                0,
                simulation_objects["core_geom"].core_length
                + simulation_objects["core_geom"].exchanger_length,
                simulation_objects["core_geom"].n_cells_core
                + simulation_objects["core_geom"].n_cells_exchanger,
            )

            source = beam_intensity * np.exp(
                -((x - beam_center) ** 2) / (2 * beam_width**2)
            )

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
            # we only save the final state because we are only interested in the steady state solution
            core_states = [simulation_objects["core_state"]]
            secondary_states = [simulation_objects["secondary_state"]]
            neutronic_states = [final_state]
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
        elif simulation_objects["neutronic_mode"] == "source_driven":
            # we first override the mode of the neutronics solver and check if the uncoupled problem is solvable
            beam_center = simulation_objects["time_params"].accelerator_center
            beam_width = simulation_objects["time_params"].accelerator_width
            beam_intensity = simulation_objects[
                "time_params"
            ].accelerator_intensity_values[0]
            # source is a Gaussian distribution of the beam intensity centered at the beam center with a width of the beam width
            x = np.linspace(
                0,
                simulation_objects["core_geom"].core_length
                + simulation_objects["core_geom"].exchanger_length,
                simulation_objects["core_geom"].n_cells_core
                + simulation_objects["core_geom"].n_cells_exchanger,
            )

            source = beam_intensity * np.exp(
                -((x - beam_center) ** 2) / (2 * beam_width**2)
            )

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
            print(
                "Steady state source driven problem solved, starting transient simulation"
            )
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
        else:
            raise ValueError("Invalid neutronics mode")
    
    # Save post-processing data
    save_post_processing(simulation_objects, core_states, neutronic_states, secondary_states)

    


if __name__ == "__main__":
    main()
