import os
import numpy as np

def save_post_processing(simulation_objects, core_states, neutronic_states, secondary_states):
    if simulation_objects["post_processing_params"]["output"]:
        output_dir = simulation_objects["post_processing_params"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = simulation_objects["post_processing_params"]["file_prefix"]
        output_starting_time = simulation_objects["post_processing_params"]["output_starting_time"]
        output_quantities = simulation_objects["post_processing_params"]["output_quantities"]
        index_to_start = (np.abs(simulation_objects["time_params"].time_values - output_starting_time)).argmin()
        
        if simulation_objects["simulation_mode"] == "transient":
            np.save(
                os.path.join(output_dir, f"{file_prefix}_time.npy"),
                simulation_objects["time_params"].time_values[index_to_start:]
            )
        
        np.save(os.path.join(output_dir, f"{file_prefix}_CORE_X.npy"), simulation_objects["core_geom"].x)
        np.save(os.path.join(output_dir, f"{file_prefix}_SECONDARY_X.npy"), simulation_objects["secondary_geom"].x)
        
        if "core" in output_quantities:
            for quantity_dict in output_quantities["core"]:
                for quantity_name, enabled in quantity_dict.items():
                    if enabled:
                        if quantity_name == "temperature":
                            core_temperature_array = np.array([state.temperature for state in core_states])
                            np.save(os.path.join(output_dir, f"{file_prefix}_CORE_TEMPERATURE.npy"), core_temperature_array)
                        elif quantity_name == "mass_flow_rate":
                            core_mass_flow_rate_array = np.array([state.flow_rate for state in core_states])
                            np.save(os.path.join(output_dir, f"{file_prefix}_CORE_MASS_FLOW_RATE.npy"), core_mass_flow_rate_array)
                        elif quantity_name == "power":
                            power_array = np.array([state.power for state in neutronic_states])
                            np.save(os.path.join(output_dir, f"{file_prefix}_POWER.npy"), power_array)
                        elif quantity_name == "neutron_flux":
                            neutron_flux_array = np.array([state.phi for state in neutronic_states])
                            np.save(os.path.join(output_dir, f"{file_prefix}_NEUTRON_FLUX.npy"), neutron_flux_array)
                        elif quantity_name == "precursor_concentration":
                            precursor_concentration_array = np.array([state.C for state in neutronic_states])
                            np.save(os.path.join(output_dir, f"{file_prefix}_PRECURSOR_CONCENTRATION.npy"), precursor_concentration_array)
        
        if "secondary_loop" in output_quantities:
            for quantity_dict in output_quantities["secondary_loop"]:
                for quantity_name, enabled in quantity_dict.items():
                    if enabled:
                        if quantity_name == "temperature":
                            secondary_temperature_array = np.array([state.temperature for state in secondary_states])
                            np.save(os.path.join(output_dir, f"{file_prefix}_SECONDARY_TEMPERATURE.npy"), secondary_temperature_array)
                        elif quantity_name == "mass_flow_rate":
                            secondary_mass_flow_rate_array = np.array([state.flow_rate for state in secondary_states])
                            np.save(os.path.join(output_dir, f"{file_prefix}_SECONDARY_MASS_FLOW_RATE.npy"), secondary_mass_flow_rate_array)