# tests/test_input_parser.py

import unittest
from parsers.input_parser import InputDeck, InputDeckModel
from pydantic import ValidationError

class TestInputParser(unittest.TestCase):

    def test_valid_input_deck(self):
        # Assume 'input/input_deck.yaml' is the valid YAML provided
        input_deck_path = 'input/input_deck.yaml'
        input_deck = InputDeck.from_yaml(input_deck_path)
        self.assertEqual(input_deck.simulation.total_time, 100.0)
        self.assertEqual(len(input_deck.operational_parameters.pump_primary.schedule), 3)
        self.assertEqual(input_deck.materials.primary_salt['density'], 2250.0)
        # Add more assertions as needed

    def test_missing_required_field(self):
        # Create a mock YAML with a missing field
        mock_data = {
            "simulation": {
                "total_time": 100.0
                # "time_step" is missing
            },
            "geometry": {
                "core_length": 2.0,
                "exchanger_length": 1.0,
                "core_radius": 1.0,
                "cooling_loop_first_length": 5.0,
                "cooling_loop_second_length": 5.0,
                "secondary_loop_radius": 1.0,
                "heat_exchanger_coefficient": 1.0,
                "n_core": 20,
                "n_exchanger": 10,
                "n_cooling_loop_first_segment": 50,
                "n_cooling_loop_second_segment": 50
            },
            "materials": {
                "primary_salt": {
                    "density": 2250.0,
                    "cp": 1967.0,
                    "expansion": 2.12e-4
                },
                "secondary_salt": {
                    "density": 2250.0,
                    "cp": 1967.0
                }
            },
            "nuclear_data": {
                "diffusion_coefficient": 1.21e-2,
                "absorption_cross_section": 3.76e-1,
                "fission_cross_section": 1.85e-1,
                "nu_fission": 2.4,
                "beta": 650e-5,
                "decay_constant": 0.1
            },
            "operational_parameters": {
                "pump_primary": {
                    "mode": "flow_rate_control",
                    "schedule": [
                        {"time": 0.0, "flow_rate": 10.0},
                        {"time": 50.0, "flow_rate": 20.0},
                        {"time": 80.0, "flow_rate": 5.0}
                    ]
                },
                "pump_secondary": {
                    "mode": "flow_rate_control",
                    "schedule": [
                        {"time": 0.0, "flow_rate": 5.0},
                        {"time": 70.0, "flow_rate": 10.0}
                    ]
                },
                "secondary_inlet_temp": {
                    "schedule": [
                        {"time": 0.0, "temperature": 400.0},
                        {"time": 60.0, "temperature": 450.0}
                    ]
                }
            }
        }
        
        with self.assertRaises(ValidationError):
            InputDeckModel(**mock_data)

    def test_invalid_type(self):
        # Create a mock YAML with an incorrect type
        mock_data = {
            "simulation": {
                "total_time": "one hundred",  # Should be float
                "time_step": 0.01
            },
            "geometry": {
                "core_length": 2.0,
                "exchanger_length": 1.0,
                "core_radius": 1.0,
                "cooling_loop_first_length": 5.0,
                "cooling_loop_second_length": 5.0,
                "secondary_loop_radius": 1.0,
                "heat_exchanger_coefficient": 1.0,
                "n_core": 20,
                "n_exchanger": 10,
                "n_cooling_loop_first_segment": 50,
                "n_cooling_loop_second_segment": 50
            },
            "materials": {
                "primary_salt": {
                    "density": 2250.0,
                    "cp": 1967.0,
                    "expansion": 2.12e-4
                },
                "secondary_salt": {
                    "density": 2250.0,
                    "cp": 1967.0
                }
            },
            "nuclear_data": {
                "diffusion_coefficient": 1.21e-2,
                "absorption_cross_section": 3.76e-1,
                "fission_cross_section": 1.85e-1,
                "nu_fission": 2.4,
                "beta": 650e-5,
                "decay_constant": 0.1
            },
            "operational_parameters": {
                "pump_primary": {
                    "mode": "flow_rate_control",
                    "schedule": [
                        {"time": 0.0, "flow_rate": 10.0},
                        {"time": 50.0, "flow_rate": 20.0},
                        {"time": 80.0, "flow_rate": 5.0}
                    ]
                },
                "pump_secondary": {
                    "mode": "flow_rate_control",
                    "schedule": [
                        {"time": 0.0, "flow_rate": 5.0},
                        {"time": 70.0, "flow_rate": 10.0}
                    ]
                },
                "secondary_inlet_temp": {
                    "schedule": [
                        {"time": 0.0, "temperature": 400.0},
                        {"time": 60.0, "temperature": 450.0}
                    ]
                }
            }
        }
        
        with self.assertRaises(ValidationError):
            InputDeckModel(**mock_data)
    
    def test_get_pump_flow_rate(self):
        input_deck_path = 'input/input_deck.yaml'
        input_deck = InputDeck.from_yaml(input_deck_path)
        
        # Test at a time within the schedule range
        flow_rate = input_deck.get_pump_flow_rate(input_deck.operational_parameters.pump_primary, 25.0)
        expected_flow_rate = 15.0  # Linear interpolation between 10.0 at t=0 and 20.0 at t=50
        self.assertAlmostEqual(flow_rate, expected_flow_rate)
        
        # Test at the start of the schedule
        flow_rate = input_deck.get_pump_flow_rate(input_deck.operational_parameters.pump_primary, 0.0)
        expected_flow_rate = 10.0
        self.assertEqual(flow_rate, expected_flow_rate)
        
        # Test at the end of the schedule
        flow_rate = input_deck.get_pump_flow_rate(input_deck.operational_parameters.pump_primary, 80.0)
        expected_flow_rate = 5.0
        self.assertEqual(flow_rate, expected_flow_rate)

    def test_get_secondary_inlet_temp(self):
        input_deck_path = 'input/input_deck.yaml'
        input_deck = InputDeck.from_yaml(input_deck_path)
        
        # Test at a time within the schedule range
        temp = input_deck.get_secondary_inlet_temp(30.0)
        expected_temp = 425.0  # Linear interpolation between 400.0 at t=0 and 450.0 at t=60
        self.assertAlmostEqual(temp, expected_temp)
        
        # Test at the start of the schedule
        temp = input_deck.get_secondary_inlet_temp(0.0)
        expected_temp = 400.0
        self.assertEqual(temp, expected_temp)
        
        # Test at the end of the schedule
        temp = input_deck.get_secondary_inlet_temp(60.0)
        expected_temp = 450.0
        self.assertEqual(temp, expected_temp)

if __name__ == '__main__':
    unittest.main()
