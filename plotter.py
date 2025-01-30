# plotter.py

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from parsers.input_parser import InputDeck

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot the simulation results.")
    parser.add_argument(
        "input_deck_path",
        type=str,
        help="Path to the input deck YAML file.",
    )
    # add an optional argument to specify the starting plot time
    parser.add_argument(
        "--starting_time",
        type=float,
        default=0.0,
        help="Starting time for the plot.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_deck_path = args.input_deck_path

    # Parse the input deck
    input_deck = InputDeck.from_yaml(input_deck_path)

    if not input_deck.post_processing.output:
        print("Post-processing output is disabled in the input deck.")
        return

    output_dir = input_deck.post_processing.output_dir
    file_prefix = input_deck.post_processing.file_prefix
    simulation_mode = input_deck.simulation.mode
    # get the core number of cells because then we can plot the outlet temperature
    n_cells_core = input_deck.geometry.n_core
    target_power = input_deck.nuclear_data.power

    if simulation_mode == "steady_state":
        # Load and plot steady state distributions
        core_temperature = np.load(os.path.join(output_dir, f"{file_prefix}_CORE_TEMPERATURE.npy"))
        core_mass_flow_rate = np.load(os.path.join(output_dir, f"{file_prefix}_CORE_MASS_FLOW_RATE.npy"))
        neutron_flux = np.load(os.path.join(output_dir, f"{file_prefix}_NEUTRON_FLUX.npy"))
        precursor_concentration = np.load(os.path.join(output_dir, f"{file_prefix}_PRECURSOR_CONCENTRATION.npy"))
        secondary_temperature = np.load(os.path.join(output_dir, f"{file_prefix}_SECONDARY_TEMPERATURE.npy"))

        core_x = np.load(os.path.join(output_dir, f"{file_prefix}_CORE_X.npy"))
        secondary_x = np.load(os.path.join(output_dir, f"{file_prefix}_SECONDARY_X.npy"))

        plt.figure()
        plt.plot(core_x, core_temperature[-1], label="Core Temperature")
        plt.xlabel("Position [m]")
        plt.ylabel("Temperature [K]")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(core_x, core_mass_flow_rate[-1], label="Core Mass Flow Rate")
        plt.xlabel("Position [m]")
        plt.ylabel("Mass Flow Rate [kg/s]")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(core_x, neutron_flux[-1], label="Neutron Flux")
        plt.xlabel("Position [m]")
        plt.ylabel("Neutron Flux [n/m^2-s]")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(core_x, precursor_concentration[-1], label="Precursor Concentration")
        plt.xlabel("Position [m]")
        plt.ylabel("Precursor Concentration")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(secondary_x, secondary_temperature[-1], label="Secondary Temperature")
        plt.xlabel("Position [m]")
        plt.ylabel("Temperature [K]")
        plt.legend()
        plt.show()

    elif simulation_mode == "transient":
        # Load and plot transient data
        start_time = args.starting_time
        time = np.load(os.path.join(output_dir, f"{file_prefix}_time.npy"))
        index_to_start = (np.abs(time - start_time)).argmin()
        power = np.load(os.path.join(output_dir, f"{file_prefix}_POWER.npy"))
        core_temperature = np.load(os.path.join(output_dir, f"{file_prefix}_CORE_TEMPERATURE.npy"))
        secondary_temperature = np.load(os.path.join(output_dir, f"{file_prefix}_SECONDARY_TEMPERATURE.npy"))
        core_mass_flow_rate = np.load(os.path.join(output_dir, f"{file_prefix}_CORE_MASS_FLOW_RATE.npy"))

        plt.figure()
        plt.plot(time[index_to_start:], (power[index_to_start:] / target_power) * 100, label="Power")
        plt.xlabel("Time [s]")
        plt.ylabel("Power [% of Target]")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(time[index_to_start:], core_temperature[index_to_start:, 0], label="Inlet Core Temperature")
        plt.plot(time[index_to_start:], core_temperature[index_to_start:, n_cells_core], label="Outlet Core Temperature")
        plt.xlabel("Time [s]")
        plt.ylabel("Temperature [K]")
        plt.legend()
        plt.show()

        mean_core_mass_flow_rate = np.mean(core_mass_flow_rate[index_to_start:], axis=1)
        plt.figure()
        plt.plot(time[index_to_start:], mean_core_mass_flow_rate, label="Mean Core Mass Flow Rate")
        plt.xlabel("Time [s]")
        plt.ylabel("Mass Flow Rate [kg/s]")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()