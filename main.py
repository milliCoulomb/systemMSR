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
from utils.time_parameters import TimeParameters
from couplers.SteadyStateCoupler import SteadyStateCoupler
from couplers.UnsteadyCoupler import UnsteadyCoupler


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
    times_primary = np.array(
        [point.time for point in input_deck.operational_parameters.pump_primary.schedule]
    )
    rates_primary = np.array(
        [point.flow_rate for point in input_deck.operational_parameters.pump_primary.schedule]
    )
    print(f"Times: {times_primary}")
    print(f"Flow Rates: {rates_primary}")
    times_secondary = np.array(
        [point.time for point in input_deck.operational_parameters.pump_secondary.schedule]
    )
    rates_secondary = np.array(
        [point.flow_rate for point in input_deck.operational_parameters.pump_secondary.schedule]
    )
    print(f"Times: {times_secondary}")
    print(f"Flow Rates: {rates_secondary}")

    print("Secondary Inlet Temperature Schedule:")
    times_secondary_inlet_temperature = np.array(
        [
            point.time
            for point in input_deck.operational_parameters.secondary_inlet_temp.schedule
        ]
    )
    temperatures_secondary_inlet = np.array(
        [
            point.temperature
            for point in input_deck.operational_parameters.secondary_inlet_temp.schedule
        ]
    )
    print(f"Times: {times_secondary_inlet_temperature}")
    print(f"Temperatures: {temperatures_secondary_inlet}")

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

    NUMBER_TIME_STEPS = int(input_deck.simulation.total_time / input_deck.simulation.time_step)
    # initiate the time parameters
    time_params = TimeParameters(
        time_step=input_deck.simulation.time_step,
        total_time=input_deck.simulation.total_time,
        num_time_steps=NUMBER_TIME_STEPS,
        pump_values_primary=rates_primary,
        time_values_primary_pump=times_primary,
        pump_values_secondary=rates_secondary,
        time_values_secondary_pump=times_secondary,
        secondary_inlet_temperature_values=temperatures_secondary_inlet,
        time_values_secondary_inlet_temperature=times_secondary_inlet_temperature,
    )
    time = time_params.time_values
    # plot the pump schedule
    plt.plot(time, time_params.pump_values_primary, label="Primary Pump")
    plt.plot(time, time_params.pump_values_secondary, label="Secondary Pump")
    plt.xlabel("Time [s]")
    plt.ylabel("Flow Rate [kg/s]")
    plt.legend()
    plt.show()

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
    coupler = SteadyStateCoupler(th_solver, neut_solver)
    core_state, secondary_state, neutronic_state = coupler.solve(
        core_state, secondary_state, final_state
    )
    # print the coupled keff
    print(f"Final keff: {neutronic_state.keff}")
    # renormalize the fission cross section by the keff
    neut_params.Sigma_f = neut_params.Sigma_f / neutronic_state.keff
    # rebuild the neutronics solver
    neut_solver = NeutronicsSolver(neut_params, fvm, core_geom)
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
    # start the plot at the time value closest to 200s
    index_to_start = (np.abs(time_params.time_values - 200)).argmin()
    time = time_params.time_values[index_to_start:]
    power = np.array([state.power for state in neutronic_states])[index_to_start:]
    # also duplicate the axis to show the mass flow rate of the secondary loop
    flow_rate_secondary = np.array([state.flow_rate for state in secondary_states])[index_to_start:]
    flow_rate_primary = np.array([state.flow_rate for state in core_states])[index_to_start:]
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Power [MW]", color=color)
    ax1.plot(time, power / 1e6, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Flow Rate [kg/s]", color=color)
    ax2.plot(time, flow_rate_secondary, color=color)
    # also plot the primary flow rate
    ax2.plot(time, flow_rate_primary, color="tab:green")
    ax2.tick_params(axis="y", labelcolor=color)
    fig.tight_layout()
    plt.show()
    plt.close()

    # okay, now I need to do an animation as a function of time
    # I will animate the temperature distribution in the core
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_ylim(800, 1200)
    ax.set_xlim(0, core_geom.core_length)
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Temperature Distribution in the Core")
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return (line,)
    # animation function. This is called sequentially
    def animate(i):
        x = np.linspace(0, core_geom.core_length, len(core_states[i + index_to_start].temperature))
        y = core_states[i + index_to_start].temperature
        line.set_data(x, y)
        return (line,)
    # call the animator
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(core_states) - index_to_start, interval=200, blit=True
    )
    plt.show()
    plt.close()




if __name__ == "__main__":
    main()
