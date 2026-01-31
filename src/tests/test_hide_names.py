"""
Test script demonstrating the option to hide layer names.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.NN_DEFINITION_UTILITIES import NeuralNetwork, FullyConnectedLayer, VectorInput, VectorOutput
from src.NN_PLOTTING_UTILITIES import plot_network, PlotConfig

# Create output directory at project root
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
output_dir = os.path.join(project_root, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

# Create a neural network
print("Creating neural network...")
nn = NeuralNetwork("Clean Network")

nn.add_layer(VectorInput(num_features=5, name="Input"))
nn.add_layer(FullyConnectedLayer(num_neurons=8, activation="relu", name="Hidden"))
nn.add_layer(VectorOutput(num_neurons=3, activation="softmax", name="Output"))

print(f"\nNetwork structure:\n{nn}\n")

# Test 1: With layer names (default)
print("Generating plot WITH layer names...")
config_with_names = PlotConfig(
    figsize=(12, 6),
    show_layer_names=True
)

plot_network(
    nn,
    title="Network with Layer Names",
    save_path=os.path.join(output_dir, "test_with_names.png"),
    show=False,
    config=config_with_names
)

print("✓ Plot saved as 'test_with_names.png'")

# Test 2: Without layer names
print("\nGenerating plot WITHOUT layer names...")
config_without_names = PlotConfig(
    figsize=(12, 6),
    show_layer_names=False
)

plot_network(
    nn,
    title="Network without Layer Names (Clean View)",
    save_path=os.path.join(output_dir, "test_without_names.png"),
    show=False,
    config=config_without_names
)

print("✓ Plot saved as 'test_without_names.png'")

print("\nComparison:")
print("  • test_with_names.png: Shows layer names below each layer")
print("  • test_without_names.png: Clean view without layer labels")
