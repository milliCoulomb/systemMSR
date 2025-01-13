// src/input/input_deck.rs
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SimulationSettings {
    pub total_time: f64,               // [s] Total simulation time
    pub time_step: f64,                // [s] Time step
    pub max_picard_iterations: usize,  // Maximum Picard iterations per time step
    pub picard_tolerance: f64,         // Tolerance for Picard convergence
}

#[derive(Debug, Deserialize)]
pub struct Geometry {
    pub core_length: f64,               // [m]
    pub recirculation_length: f64,      // [m]
    pub secondary_length: f64,          // [m]
    pub n_cells_core: usize,            // Number of cells in core
    pub n_cells_recirc: usize,          // Number of cells in recirculation loop
    pub n_cells_secondary: usize,       // Number of cells in secondary loop
}

#[derive(Debug, Deserialize)]
pub struct Neutronics {
    pub diffusion_coefficient: Vec<f64>, // [m^2/s] per cell
    pub absorption_xs: Vec<f64>,         // [1/m] per cell
    pub fission_xs: Vec<f64>,            // [1/m] per cell
    pub nu: f64,                          // Average neutrons/fission
    pub velocity: f64,                    // [m/s]
    pub delayed_neutron_fraction: f64,    // Fraction of delayed neutrons
}

#[derive(Debug, Deserialize)]
pub struct FluidProperties {
    pub density: f64,                    // [kg/m^3]
    pub heat_capacity: f64,              // [J/(kg*K)]
    pub viscosity: f64,                  // [Pa*s]
    pub thermal_conductivity: f64,       // [W/(m*K)]
    pub reference_temperature: f64,      // [K]
}

#[derive(Debug, Deserialize)]
pub struct ThermalHydraulics {
    pub primary_fluid: FluidProperties,
    pub secondary_fluid: FluidProperties,
}

#[derive(Debug, Deserialize)]
pub struct PumpSchedule {
    pub time: Vec<f64>,                   // [s]
    pub flow_rate: Vec<f64>,              // [m^3/s] or [kg/s]
}

#[derive(Debug, Deserialize)]
pub struct OperationalParameters {
    pub primary_pump_table: PumpSchedule,
    pub secondary_pump_table: PumpSchedule,
    pub secondary_inlet_temperature_table: PumpSchedule, // Using PumpSchedule for temperature table
}

#[derive(Debug, Deserialize)]
pub struct OutputSettings {
    pub write_interval: f64,     // [s]
    pub output_folder: String,
}

#[derive(Debug, Deserialize)]
pub struct InputDeck {
    pub simulation: SimulationSettings,
    pub geometry: Geometry,
    pub neutronics: Neutronics,
    pub thermal_hydraulics: ThermalHydraulics,
    pub operational_parameters: OperationalParameters,
    pub output: OutputSettings,
}