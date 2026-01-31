"""
Quick test to verify variable names are displaying.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput, VectorOutput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

# Create a simple network
nn = NeuralNetwork("Test Network")

input_layer = VectorInput(num_features=4, name="Input")
nn.add_layer(input_layer)

hidden = FullyConnectedLayer(num_neurons=6, activation="relu", name="Hidden")
nn.add_layer(hidden, parent_ids=[input_layer.layer_id])

output = VectorOutput(num_neurons=3, activation="softmax", name="Output")
nn.add_layer(output, parent_ids=[hidden.layer_id])

# Configure with variable names
config = PlotConfig(
    figsize=(12, 6),
    show_title=True,
    show_layer_names=True,
    background_color='white',
    
    # Variable names - TESTING
    layer_variable_names={
        'Input': 'TEST INPUT LABEL',
        'Output': 'TEST OUTPUT LABEL'
    },
    show_layer_variable_names=True,
    layer_variable_names_fontsize=14,
    layer_variable_names_position='side',
    
    # Add boxes to make it more visible
    layer_styles={
        'Input': LayerStyle(
            neuron_fill_color='#FFD700',
            box_around_layer=True,
            box_fill_color='#FFFACD',
            box_edge_color='#B8860B'
        ),
        'Output': LayerStyle(
            neuron_fill_color='#90EE90',
            box_around_layer=True,
            box_fill_color='#E6FFE6',
            box_edge_color='#228B22'
        )
    }
)

print("Generating test figure with variable names...")
print(f"Configuration:")
print(f"  - layer_variable_names: {config.layer_variable_names}")
print(f"  - show_layer_variable_names: {config.show_layer_variable_names}")
print(f"  - layer_variable_names_position: {config.layer_variable_names_position}")

output_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(output_dir, exist_ok=True)

test_path = os.path.join(output_dir, "test_variable_names.png")
plot_network(nn, config=config, save_path=test_path, show=False)

print(f"\n✓ Test figure saved: {test_path}")
print("\nPlease check if the variable names 'TEST INPUT LABEL' and 'TEST OUTPUT LABEL' appear in the figure.")
