// src/input/parser.rs

use serde_yaml;
use std::fs::File;
use std::io::Read;
use crate::input::InputDeck;

/// Parses the input deck from a YAML file.
///
/// # Arguments
///
/// * `file_path` - Path to the YAML input file.
///
/// # Returns
///
/// * `Ok(InputDeck)` if parsing is successful.
/// * `Err` if an error occurs during file reading or parsing.
pub fn parse_input_deck(file_path: &str) -> Result<InputDeck, Box<dyn std::error::Error>> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let input_deck: InputDeck = serde_yaml::from_str(&contents)?;
    Ok(input_deck)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_parse_input_deck_success() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("valid_input.yaml");
        let mut file = File::create(&file_path).unwrap();
        let yaml_content = r#"
simulation:
  total_time: 100.0
  time_step: 0.1
  max_picard_iterations: 10
  picard_tolerance: 1e-5
geometry:
  core_length: 10.0
  recirculation_length: 5.0
  secondary_length: 3.0
  n_cells_core: 100
  n_cells_recirc: 50
  n_cells_secondary: 30
neutronics:
  diffusion_coefficient: [1.0, 1.0, 1.0]
  absorption_xs: [0.1, 0.1, 0.1]
  fission_xs: [0.01, 0.01, 0.01]
  nu: 2.5
  velocity: 1.5
  delayed_neutron_fraction: 0.007
thermal_hydraulics:
  primary_fluid:
    density: 1000.0
    heat_capacity: 4184.0
    viscosity: 0.001
    thermal_conductivity: 0.6
    reference_temperature: 300.0
  secondary_fluid:
    density: 800.0
    heat_capacity: 4000.0
    viscosity: 0.0012
    thermal_conductivity: 0.5
    reference_temperature: 290.0
operational_parameters:
  primary_pump_table:
    time: [0.0, 50.0, 100.0]
    flow_rate: [1.0, 1.5, 1.0]
  secondary_pump_table:
    time: [0.0, 50.0, 100.0]
    flow_rate: [0.8, 1.2, 0.8]
  secondary_inlet_temperature_table:
    time: [0.0, 50.0, 100.0]
    flow_rate: [290.0, 295.0, 290.0]
output:
  write_interval: 10.0
  output_folder: "outputs"
"#;
        file.write_all(yaml_content.as_bytes()).unwrap();

        let result = parse_input_deck(file_path.to_str().unwrap());
        assert!(result.is_ok());
        let input_deck = result.unwrap();
        assert_eq!(input_deck.simulation.total_time, 100.0);
    }

    #[test]
    fn test_parse_input_deck_failure() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("invalid_input.yaml");
        let mut file = File::create(&file_path).unwrap();
        let invalid_yaml = "invalid: [unbalanced brackets";
        file.write_all(invalid_yaml.as_bytes()).unwrap();

        let result = parse_input_deck(file_path.to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_input_deck_file_not_found() {
        let result = parse_input_deck("non_existent_file.yaml");
        assert!(result.is_err());
    }
}