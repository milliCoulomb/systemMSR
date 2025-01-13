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