# main.py

from parsers.input_parser import InputDeck

def main():
    # Path to your input deck
    input_deck_path = 'input/input_deck.yaml'
    
    # Parse the input deck
    input_deck = InputDeck.from_yaml(input_deck_path)
    
    # Accessing some parameters as examples
    print(f"Total Simulation Time: {input_deck.simulation.total_time} s")
    print(f"Time Step: {input_deck.simulation.time_step} s")
    
    print(f"Core Length: {input_deck.geometry.core_length} m")
    print(f"Heat Exchanger Coefficient: {input_deck.geometry.heat_exchanger_coefficient} W/m^3-K")
    
    print(f"Primary Salt Density: {input_deck.materials.primary_salt['density']} kg/m^3")
    print(f"Secondary Salt CP: {input_deck.materials.secondary_salt['cp']} J/kg-K")
    
    print(f"Nuclear Diffusion Coefficient: {input_deck.nuclear_data.diffusion_coefficient} m")
    
    # Example: Access pump primary schedule
    print("Primary Pump Schedule:")
    for point in input_deck.operational_parameters.pump_primary.schedule:
        print(f"  Time: {point.time} s, Flow Rate: {point.flow_rate} kg/s")
    
    # Similarly, access other parameters as needed
    
    # Proceed with the rest of your simulation setup
    # ...

if __name__ == "__main__":
    main()
