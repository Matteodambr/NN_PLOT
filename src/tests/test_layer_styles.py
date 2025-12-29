"""
Test script for layer-specific styling in neural network plots.
Demonstrates custom colors, edge styles, and line widths for each layer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig, LayerStyle

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

# Create a simple neural network
print("Creating neural network with custom layer styles...")
nn = NeuralNetwork("Styled Network")

# Add layers with descriptive names
nn.add_layer(FullyConnectedLayer(num_neurons=4, name="Input"))
nn.add_layer(FullyConnectedLayer(num_neurons=6, activation="relu", name="Hidden1"))
nn.add_layer(FullyConnectedLayer(num_neurons=5, activation="relu", name="Hidden2"))
nn.add_layer(FullyConnectedLayer(num_neurons=3, activation="softmax", name="Output"))

print(f"\nNetwork structure:\n{nn}\n")

# Create custom styles for each layer
print("Setting up layer-specific styles...")

layer_styles = {
    "Input": LayerStyle(
        neuron_fill_color="lightcoral",
        neuron_edge_color="darkred",
        neuron_edge_width=2.0,
        connection_linewidth=1.5,
        connection_color="red",
        connection_alpha=0.6
    ),
    "Hidden1": LayerStyle(
        neuron_fill_color="lightgreen",
        neuron_edge_color="darkgreen",
        neuron_edge_width=2.5,
        connection_linewidth=1.0,
        connection_color="green",
        connection_alpha=0.5
    ),
    "Hidden2": LayerStyle(
        neuron_fill_color="lightyellow",
        neuron_edge_color="orange",
        neuron_edge_width=2.0,
        connection_linewidth=0.8,
        connection_color="orange",
        connection_alpha=0.4
    ),
    "Output": LayerStyle(
        neuron_fill_color="lightblue",
        neuron_edge_color="darkblue",
        neuron_edge_width=3.0,
        # Output layer has no outgoing connections
    )
}

# Create plot configuration with layer styles
config = PlotConfig(
    figsize=(14, 8),
    layer_styles=layer_styles,
    show_layer_names=True,
    neuron_radius=0.4,
    layer_spacing=3.5
)

# Plot the network
print("Generating plot with custom layer styles...")
plot_network(
    nn,
    title="Neural Network with Custom Layer Styles",
    save_path=os.path.join(output_dir, "test_layer_styles.png"),
    show=False,
    config=config
)

print("✓ Plot saved as 'test_layer_styles.png'")
print("\nLayer styling features demonstrated:")
print("  • Input layer: Red neurons with red connections")
print("  • Hidden1 layer: Green neurons with green connections")
print("  • Hidden2 layer: Yellow neurons with orange connections")
print("  • Output layer: Blue neurons with thicker edges")
print("  • Each layer has different edge widths and connection line widths")
